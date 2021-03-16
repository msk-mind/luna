'''
Created on November 30, 2020

@author: aukermaa@mskcc.org

Memorial Sloan Kettering Cancer Center 
'''

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.common.utils import generate_uuid
from data_processing.common.Container import Container 
from data_processing.common.Container import Node 

from pyspark.sql.functions import udf, lit, col, array
from pyspark.sql.types import StringType, MapType

import shutil, sys, importlib, glob, yaml, os, subprocess, time
from pathlib import Path

import openslide 
import requests 

logger = init_logger()

TRY_S3=False
SCHEMA_FILE='data_ingestion_template_schema.yml'
DATA_CFG = 'DATA_CFG'
APP_CFG = 'APP_CFG'

def sql_cleaner(s):
    return s.replace(".","_").replace(" ","_")

def parse_slide_id(path):
    """ Slide stem is their slide id """
    posix_file_path = path.split(':')[-1]

    slide_id = Path(posix_file_path).stem

    return slide_id 

def parse_openslide(path):
    """ From https://github.com/msk-mind/sandbox/blob/master/pathology/slide_to_proxy.py """
    """ Parse openslide header information """

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
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(data_config_file, app_config_file, process_string):
    """
    This module generates a set of proxy tables for pathology data based on information specified in the template file.

    Example:
        python -m data_processing.pathology.proxy_table.generate \
        --template_file {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
        --process_string transfer,delta

    """
    with CodeTimer(logger, 'generate proxy table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data_ingestions_template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name="APP_CFG", config_file=app_config_file)
        cfg = ConfigSet(name="DATA_CFG", config_file=data_config_file, schema_file=SCHEMA_FILE)

        # write template file to manifest_yaml under LANDING_PATH
        # todo: write to hdfs without using local gpfs/
        hdfs_path = os.environ['MIND_GPFS_DIR']
        landing_path = cfg.get_value(path='DATA_CFG::LANDING_PATH')
        
        full_landing_path = os.path.join(hdfs_path, landing_path)     
        if not os.path.exists(full_landing_path):
            os.makedirs(full_landing_path)
        shutil.copy(data_config_file, os.path.join(full_landing_path, "manifest.yaml"))
        logger.info("template file copied to %s", os.path.join(full_landing_path, "manifest.yaml"))

        # subprocess - create proxy table
        if 'delta' in processes or 'all' in processes:
            exit_code = create_proxy_table(app_config_file)
            if exit_code != 0:
                logger.error("Delta table creation had errors. Exiting.")
                return
        if 'graph' in processes or 'all' in processes:
            exit_code = update_graph(app_config_file)
            if exit_code != 0:
                logger.error("Graph creation had errors. Exiting.")
                return


def create_proxy_table(config_file):

    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="data_processing.pathology.proxy_table.generate")

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
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="data_processing.pathology.proxy_table.generate")

    table_path = const.TABLE_LOCATION(cfg)

    namespace            = cfg.get_value("DATA_CFG::PROJECT")
    api_base_url         = cfg.get_value("APP_CFG::api_base_url")
    cohort_service_host  = cfg.get_value("APP_CFG::cohortManager_host")
    cohort_service_port  = cfg.get_value("APP_CFG::cohortManager_port")
    cohort_uri           = f"http://{cohort_service_host}:{cohort_service_port}{api_base_url}"

    logger.info ("Requesting %s, %s", os.path.join(cohort_uri, "cohort", namespace), requests.put(os.path.join(cohort_uri, "cohort", namespace)).text)

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        logger.info("Loading %s:", table_path)
        tuple_to_add = spark.read.format("delta").load(table_path).select("slide_id", "path", "metadata").toPandas()

    with CodeTimer(logger, 'synchronize lake'):
        container = Container( cfg ).setNamespace(namespace)
        for _, row in tuple_to_add.iterrows():
            logger.info ("Requesting %s, %s", os.path.join(cohort_uri, "container", "slide", row.slide_id), requests.put(os.path.join(cohort_uri, "container", "slide", row.slide_id)).text)
            container.lookupAndAttach(row.slide_id)
            properties = row.metadata
            properties['file'] = row.path.split(':')[-1]
            node = Node("wsi", "whole_slide_image", properties)
            container.add(node)
        container.saveAll()

    return exit_code

if __name__ == "__main__":
    cli()
