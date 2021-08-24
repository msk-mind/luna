'''
Created on November 30, 2020

@author: aukermaa@mskcc.org

Memorial Sloan Kettering Cancer Center
'''

import click

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
import luna.common.constants as const
from luna.common.utils import generate_uuid, get_absolute_path
from luna.common.DataStore import DataStore
from luna.common.DataStore import Node

from pyspark.sql.functions import udf, lit, col, array
from pyspark.sql.types import StringType, MapType

import shutil, sys, importlib, glob, yaml, os, subprocess, time
from pathlib import Path

import openslide
import requests

logger = init_logger()

TRY_S3=False
SCHEMA_FILE=get_absolute_path(__file__, 'data_ingestion_template_schema.yml')
DATA_CFG = 'DATA_CFG'
APP_CFG = 'APP_CFG'

def sql_cleaner(s):
    """Replace period and space with underscore

    Args:
        s (string): string to be cleaned

    Returns:
        string: cleaned sql string
    """
    return s.replace(".","_").replace(" ","_")

def parse_slide_id(path):
    """Parse slide id from path. Slide id is simply the name of the file without file extension

    Args:
        path (string): path to slide image

    Returns:
        string: slide id
    """
    posix_file_path = path.split(':')[-1]

    slide_id = Path(posix_file_path).stem

    return slide_id

def parse_openslide(path):
    """Extract openslide properties

    Args:
        path (string): path to slide image

    Returns:
        dict: slide metadata
    """
    posix_file_path = path.split(':')[-1]

    with openslide.OpenSlide(posix_file_path) as slide_os_handle:
        kv = dict(slide_os_handle.properties)

    meta = {}
    for key in kv.keys():
        meta[sql_cleaner(key)] = kv[key]
    return meta


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(data_config_file, app_config_file, process_string):
    """This module generates a delta table with pathology data

    This module converts the point annotation jsons in the proxy table to geojson format.
    For more details on point annotation json table, please see `point_annotation/proxy_table/generate.py`

    The configuration files are copied to your project/configs/table_name folder
    to persist the metadata used to generate the proxy table.

    INPUT PARAMETERS

    app_config_file - path to yaml file containing application runtime parameters. See config.yaml.template

    data_config_file - path to yaml file containing data input and output parameters. See data_config.yaml.template

    process_string - comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all

    TABLE SCHEMA

    - slide_id: slide id extracted from the filename. primary id for the table

    - path: path to WSI file

    - metadata: properties extracted from the slide using OpenSlide

    - wsi_record_uuid: hash of wsi file, format: WSI-{json_hash}

    - modificationTime: modification time of the wsi on disc

    - length: length of the file
    """
    with CodeTimer(logger, 'generate proxy table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data_ingestions_template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name="APP_CFG", config_file=app_config_file)
        cfg = ConfigSet(name="DATA_CFG", config_file=data_config_file, schema_file=SCHEMA_FILE)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        # subprocess - create proxy table
        if 'delta' in processes or 'all' in processes:
            exit_code = create_proxy_table()
            if exit_code != 0:
                logger.error("Delta table creation had errors. Exiting.")
                return
        if 'graph' in processes or 'all' in processes:
            exit_code = update_graph(app_config_file)
            if exit_code != 0:
                logger.error("Graph creation had errors. Exiting.")
                return


def create_proxy_table():
    """Create a proxy table with WSI metadata

    Returns:
        int: exit code 0 if successful
    """

    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="luna.pathology.proxy_table.generate")

    logger.info("generating binary proxy table... ")
    save_path = const.TABLE_LOCATION(cfg)
    logger.info ("Writing to %s", save_path)

    with CodeTimer(logger, 'load wsi metadata'):

        search_path = os.path.join(cfg.get_value(path='DATA_CFG::SOURCE_PATH'), ("**" + cfg.get_value(path='DATA_CFG::FILE_TYPE')))
        logger.debug(search_path)
        for path in glob.glob(search_path, recursive=True):
            logger.debug(path)

        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
        generate_uuid_udf   = udf(generate_uuid,   StringType())
        parse_slide_id_udf  = udf(parse_slide_id,  StringType())
        parse_openslide_udf = udf(parse_openslide, MapType(StringType(), StringType()))

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*."+cfg.get_value(path='DATA_CFG::FILE_TYPE')). \
            option("recursiveFileLookup", "true"). \
            load(cfg.get_value(path='DATA_CFG::SOURCE_PATH')). \
            drop("content").\
            withColumn("wsi_record_uuid", generate_uuid_udf  (col("path"), array(lit("WSI")))).\
            withColumn("slide_id",        parse_slide_id_udf (col("path"))).\
            withColumn("metadata",        parse_openslide_udf(col("path")))

    # parse all dicoms and save
    df.printSchema()
    df.coalesce(48).write.format("delta").save(save_path)

    processed_count = df.count()
    logger.info("Processed {} whole slide images out of total {} files".format(processed_count,cfg.get_value(path='DATA_CFG::FILE_COUNT')))
    df.show()
    return exit_code

def update_graph(config_file):
    """
    This function reads a delta table and:
        1. Creates/validates a cohort-namespace exists given the PROJECT name
        2. Creates slide containers for each slide_id, and
        3. Commits associated metadata/raw data to neo4j/minio
    """

    exit_code = 0
    cfg  = ConfigSet()
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="luna.pathology.proxy_table.generate")

    namespace  = cfg.get_value("DATA_CFG::PROJECT")
    table_path = const.TABLE_LOCATION(cfg)

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        logger.info("Loading %s:", table_path)
        tuple_to_add = spark.read.format("delta").load(table_path).select("slide_id", "path", "metadata").toPandas()

    with CodeTimer(logger, 'synchronize lake'):
        container = DataStore( cfg )
        container.createNamespace( namespace )
        container.setNamespace( namespace )
        for _, row in tuple_to_add.iterrows():
            container.createDatastore(row.slide_id, "slide")
            container.setDatastore(row.slide_id)
            properties = row.metadata
            node = Node("WholeSlideImage", "pathology.etl", properties)
            node.set_data( row.path.split(':')[-1] )
            container.put(node)


    return exit_code

if __name__ == "__main__":
    cli()
