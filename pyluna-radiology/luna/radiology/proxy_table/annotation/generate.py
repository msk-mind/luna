'''
Created on December 11, 2020

@author: rosed2@mskcc.org
'''
import os
import shutil

import click

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
import luna.common.constants as const

from pyspark.sql.functions import udf, lit, array, col
from pyspark.sql.types import StringType, MapType

from medpy.io import load

from luna.common.utils import get_absolute_path

logger = init_logger()


def parse_accession_number(path):
    # expected path: accession_number/some.mha
    return path.split("/")[-2]

def parse_metadata(path):
    # parse mha/mhd metadata
    posix_file_path = path.split(':')[-1]
 
    data, header = load(posix_file_path)
    img = header.get_sitkimage()

    # some metadata from mha/mhd
    kv = {}
    kv["origin"] = str(img.GetOrigin())
    kv["direction"] = str(img.GetDirection())
    kv["size"] = str(img.GetSize())
    kv["spacing"] = str(img.GetSpacing())
    
    return kv


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(data_config_file, app_config_file, process_string):
    """
        This module generates a delta table with radiology annotation data based on the (mha or mhd) input and output
         parameters specified in the data_config_file.

        Example:
            python3 -m luna.radiology.proxy_table.annotation.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file> \
                     --process_string delta
    """
    with CodeTimer(logger, 'generate proxy table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data_ingestions_template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        # TODO add transfer logic when we establish a standard method for scan annotations.

        # subprocess - create proxy table
        if 'delta' in processes or 'all' in processes:
            exit_code = create_proxy_table(data_config_file)
            if exit_code != 0:
                logger.error("Delta table creation had errors. Exiting.")
                return


def create_proxy_table(data_config):

    exit_code = 0

    spark = SparkConfig().spark_session(config_name=const.APP_CFG, 
        app_name="luna.radiology.proxy_table.annotation.generate")

    logger.info("generating proxy table... ")

    cfg = ConfigSet()

    table_path = const.TABLE_LOCATION(cfg)

    with CodeTimer(logger, 'load scan annotations'):

        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
 
        spark.sparkContext.addPyFile(
            get_absolute_path(__file__, "../../../../../pyluna-common/luna/common/utils.py"))
        from utils import generate_uuid
        generate_uuid_udf = udf(generate_uuid, StringType())
        parse_accession_number_udf = udf(parse_accession_number, StringType())
        parse_metadata_udf = udf(parse_metadata, MapType(StringType(), StringType()))

        file_types = cfg.get_value(path=const.DATA_CFG+'::FILE_TYPE')
        # if multiple file types are provided, union all DFs
        if isinstance(file_types, list) and len(file_types) > 1:
            df = spark.read.format("binaryFile") \
                .option("pathGlobFilter", file_types[0]) \
                .option("recursiveFileLookup", "true") \
                .load(cfg.get_value(path=const.DATA_CFG+'::RAW_DATA_PATH')) \
                .drop("content")

            for file_type in file_types[1:]:
                df_add = spark.read.format("binaryFile") \
                    .option("pathGlobFilter", file_type) \
                    .option("recursiveFileLookup", "true") \
                    .load(cfg.get_value(path=const.DATA_CFG+'::RAW_DATA_PATH')) \
                    .drop("content")

                df = df.union(df_add)
        else:
            df = spark.read.format("binaryFile") \
                .option("pathGlobFilter", file_types) \
                .option("recursiveFileLookup", "true") \
                .load(cfg.get_value(path=const.DATA_CFG+'::RAW_DATA_PATH')) \
                .drop("content")

        df = df.withColumn("accession_number", parse_accession_number_udf(df.path))

        # optional join with metadata csv
        if cfg.has_value(path=const.DATA_CFG+'::METADATA_CSV') \
                and cfg.has_value(path=const.DATA_CFG+'::METADATA_COLUMNS') \
                and cfg.has_value(path=const.DATA_CFG+'::METADATA_JOIN_ON'):

            # copy csv to configs
            shutil.copy(cfg.get_value(path=const.DATA_CFG+'::METADATA_CSV'),
                        const.CONFIG_LOCATION(cfg))

            # TODO: once radiology segmentation information is on REDCap join with that table.
            annotation_metadata = spark.read.load(cfg.get_value(path=const.DATA_CFG+'::METADATA_CSV'),
                                                  format="csv", header="true", inferSchema="true")

            annotation_metadata = annotation_metadata.select(cfg.get_value(path=const.DATA_CFG+'::METADATA_COLUMNS'))

            df = df.join(annotation_metadata,
                         on=cfg.get_value(path=const.DATA_CFG+'::METADATA_JOIN_ON'),
                         how="left")

        df = df.dropDuplicates(["path"]) \
            .withColumn("scan_annotation_record_uuid", generate_uuid_udf(df.path, array([lit("SCAN_ANNOTATION")]))) \
            .withColumn("metadata", parse_metadata_udf(df.path))

    # parse all dicoms and save
    df.printSchema()

    df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')) \
        .write.format("delta") \
        .mode("overwrite") \
        .save(table_path)

    # validation
    processed_count = df.count()
    logger.info("Processed {} scan annotations out of total {} files".format(
        processed_count,cfg.get_value(path=const.DATA_CFG+'::FILE_COUNT')))

    if processed_count != int(cfg.get_value(path=const.DATA_CFG+'::FILE_COUNT')):
        exit_code = 1
    return exit_code



if __name__ == "__main__":
    cli()
