'''
Created on January 30, 2021

@author: pashaa@mskcc.org
'''
import pathlib
import shutil

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const
from data_processing.pathology.common.build_geojson_from_bitmap import build_geojson_from_bitmap_pandas, concatenate_regional_geojsons_pandas
from data_processing.pathology.common.utils import get_add_triple_str

from pyspark.sql.functions import udf, lit, col, first, last, desc
from pyspark.sql.window import Window
import pandas as pd
import numpy as np

import yaml, os

from data_processing.pathology.proxy_table.annotation.slideviewer import fetch_slide_ids

logger = init_logger()

DATA_SCHEMA_FILE = os.path.join(
                      pathlib.Path(__file__).resolve().parent,
                      'data_config_schema.yml')

def create_proxy_table(app_config_file, data_config_file):
    '''
    Creates the pathology annotations proxy table with information contained in the specified data_config_file

    :param data_config_file: data configuration
    :param app_config_file: app configuration
    :return: exit_code = 0 if successful, > 0 if unsuccessful
    '''
    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(
                                 config_name=const.APP_CFG,
                                 app_name="data_processing.pathology.proxy_table.annotation.generate")

    slides = fetch_slide_ids()
    df = pd.DataFrame(data=np.array(slides), columns=["slideviewer_path", "slide_id"])


    logger.info("generating pathology annotation proxy table... ")
    wsi_path = const.TABLE_LOCATION(cfg)
    write_uri = os.environ["HDFS_URI"]
    save_path = os.path.join(write_uri, wsi_path)

    with CodeTimer(logger, 'load wsi metadata'):
        search_path = os.path.join(cfg.get_value(path=DATA_CFG + '::SOURCE_PATH'),
                                   ("**" + cfg.get_value(path=DATA_CFG + '::FILE_TYPE')))
        logger.debug(search_path)
        for path in glob.glob(search_path, recursive=True):
            logger.debug(path)

        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
        generate_uuid_udf = udf(generate_uuid, StringType())
        parse_slide_id_udf = udf(parse_slide_id, StringType())
        parse_openslide_udf = udf(parse_openslide, MapType(StringType(), StringType()))

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*." + cfg.get_value(path=DATA_CFG + '::FILE_TYPE')). \
            option("recursiveFileLookup", "true"). \
            load(cfg.get_value(path=DATA_CFG + '::SOURCE_PATH')). \
            drop("content"). \
            withColumn("wsi_record_uuid", generate_uuid_udf(col("path"))). \
            withColumn("slide_id", parse_slide_id_udf(col("path"))). \
            withColumn("metadata", parse_openslide_udf(col("path")))

    # parse all dicoms and save
    df.printSchema()
    df.coalesce(48).write.format("delta").save(save_path)

    processed_count = df.count()
    logger.info("Processed {} whole slide images out of total {} files".format(processed_count, cfg.get_value(
        path=const.DATA_CFG + '::FILE_COUNT')))
    return exit_code


@click.command()
@click.option('-d',
              '--data_config_file',
              default=None,
              type=click.Path(exists=True),
              help="path to data config file containing information required for pathology proxy data ingestion. "
                   "See data_processing/pathology/proxy_table/annotation/data_config.yaml.template")
@click.option('-a',
              '--app_config_file',
              default='config.yaml',
              type=click.Path(exists=True),
              help="path to app config file containing application configuration. See config.yaml.template")
def cli(data_config_file, app_config_file):
    '''
    This module performs the following sequence of operations -
    1) Bitmap regional pathology tissue annotations are downloaded from SlideViewer
    2) The downloaded bitmap annotations are then converted into npy arrays
    3) A proxy table is built with the following fields.

    slideviewer_path - path to original slide image in slideviewer platform
    slide_id - synonymous with image_id
    sv_project_id - same as the project_id from the data_config.yaml,refers to the SlideViewer project number.
    bmp_filepath - file path to downloaded bmp annotation file
    annotator - id of annotator for a given annotation
    date_added - date annotation first added
    date_updated - date annotation most recently updated
    bmp_record_uuid - hash of bmp annotation file, format: SVBMP-{bmp_hash}
    npy_filepath - file path to generated npy annotation file

    Usage:
    python3 -m data_processing.pathology.proxy_table.annotation.generate \
        -d {data_config_yaml} \
        -a {app_config_yaml}
    '''
    with CodeTimer(logger, 'generate regional annotation proxy table'):
        # read and validate configs
        logger.info('data config: ' + data_config_file)
        logger.info('app config: ' + app_config_file)

        data_cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file, schema_file=DATA_SCHEMA_FILE)
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)

        # write template file to manifest_yaml under LANDING_PATH
        # todo: write to hdfs without using local gpfs/
        hdfs_path = os.environ['MIND_GPFS_DIR']
        landing_path = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')

        full_landing_path = os.path.join(hdfs_path, landing_path)
        if not os.path.exists(full_landing_path):
            os.makedirs(full_landing_path)
        shutil.copy(data_config_file, os.path.join(full_landing_path, "manifest.yaml"))
        logger.info("template file copied to", os.path.join(full_landing_path, "manifest.yaml"))

        # create proxy table
        exit_code = create_proxy_table(app_config_file, data_config_file)
        if exit_code != 0:
            logger.error("Delta table creation had errors. Exiting.")
            return


if __name__ == "__main__":
    cli()