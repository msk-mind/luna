'''
Created on December 11, 2020

@author: rosed2@mskcc.org
'''
import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.utils import generate_uuid
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit, array, col
from pyspark.sql.types import StringType, MapType

from medpy.io import load

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
              help="path to yaml template file containing information required for scan annotation ingestion."
                   "See data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(data_config_file, app_config_file, process_string):
    """
    This module generates annotation tables for radiology data based on information specified in the template file.
    This supports MHA and MHD as the raw files. Specify mha or mhd as the file_type and data_type in the ingestion template. 

    Example:
        python -m data_processing.radiology.proxy_table.annotation.generate \
        --data_config_file {PATH_TO_DATA_CONFIG_FILE} \
        --app_config_file {PATH_TO_APP_CONFIG_FILE}
        --process_string delta

    """
    with CodeTimer(logger, 'generate proxy table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data_ingestions_template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)

        # TODO add transfer logic when we establish a standard method for scan annotations.

        # subprocess - create proxy table
        if 'delta' in processes or 'all' in processes:
            exit_code = create_proxy_table(data_config_file)
            if exit_code != 0:
                logger.error("Delta table creation had errors. Exiting.")
                return


def create_proxy_table(data_config):

    exit_code = 0
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config)

    spark = SparkConfig().spark_session(config_name=const.APP_CFG, 
        app_name="data_processing.radiology.proxy_table.annotation.generate")

    logger.info("generating proxy table... ")

    table_path = const.TABLE_LOCATION(cfg)

    with CodeTimer(logger, 'load scan annotations'):

        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        generate_uuid_udf = udf(generate_uuid, StringType())
        parse_accession_number_udf = udf(parse_accession_number, StringType())
        parse_metadata_udf = udf(parse_metadata, MapType(StringType(), StringType()))

        df = spark.read.format("binaryFile") \
            .option("pathGlobFilter", "*." + cfg.get_value(path=const.DATA_CFG+'::FILE_TYPE')) \
            .option("recursiveFileLookup", "true") \
            .load(cfg.get_value(path=const.DATA_CFG+'::RAW_DATA_PATH')) \
            .drop("content")

        df = df.withColumn("accession_number", parse_accession_number_udf(df.path))

        # TODO: once radiology segmentation information is on REDCap join with that table.
        # for now, we join with a curated csv that includes AccessionNumber, SeriesNumber columns
        annotation_metadata = spark.read.load(cfg.get_value(path=const.DATA_CFG+'::METADATA_CSV'),
                                              format="csv", header="true", inferSchema="true")
        annotation_metadata = annotation_metadata.select(col("AccessionNumber").alias("accession_number"),
                                                         col("SeriesNumber").alias("series_number"))

        df = df.join(annotation_metadata, on="accession_number", how="left")

        df = df.withColumn("scan_annotation_record_uuid", generate_uuid_udf(df.path, array([lit("SCAN_ANNOTATION")]))) \
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
