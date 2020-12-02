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
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit, col
from pyspark.sql.types import StringType, MapType
from pyspark import SparkFiles

import pydicom
import time
from io import BytesIO
import shutil, sys, importlib, glob
import yaml, os
import subprocess
from filehash import FileHash

from minio import Minio
from minio.error import ResponseError
 
logger = init_logger()

TRY_S3=False
SCHEMA_FILE='data_ingestion_template_schema.yml'
DATA_CFG = 'DATA_CFG'
APP_CFG = 'APP_CFG'

def generate_uuid(path):
    logger.info("Processing " + path)

    file_path = path.split(':')[-1]

    rec_hash = FileHash('sha256').hash_file(file_path)

    record_uuid = f'WSI-{rec_hash}'

    return record_uuid


def parse_openslide(path):

    dirs, filename  = os.path.split(path)
 
    kv = {}

    kv['image_id'] = os.path.basename(filename) 

    return kv

def upload_to_s3(path, bucket):

    posix_file_path = path.split(':')[-1]
    dirs, filename =  os.path.split(path)
    mock_path = os.path.join (SparkFiles.getRootDirectory(), filename)
    with open(mock_path, "w") as f:

        f.write("Hi, I'm pretending to be a wholeslide image\nOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

    minioClient = Minio(
        'dlliskimind1.mskcc.org:9001',
        access_key='mind',
        secret_key='mskcc123',
        secure=False,
    )
    print ("calling minioClient.fput_object with ", bucket, filename, mock_path)
    minioClient.fput_object(bucket, filename, mock_path) 
    store_result = minioClient.stat_object(bucket, filename).metadata
    return store_result 

@click.command()
@click.option('-t', '--template_file', default=None, type=click.Path(exists=True),
              help="path to yaml template file containing information required for pathology proxy data ingestion. "
                   "See data_processing/pathology/proxy_table/data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(template_file, config_file, process_string):
    """
    This module generates a set of proxy tables for pathology data based on information specified in the tempalte file.

    Example:
        python -m data_processing.pathology.proxy_table.generate \
        --template_file {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
        --process_string transfer,delta

    """
    with CodeTimer(logger, 'generate proxy table'):
        processes = process_string.lower().strip().split(",")
        logger.info('data_ingestions_template: ' + template_file)
        logger.info('config_file: ' + config_file)
        logger.info('processes: ' + str(processes))

        # load configs
        cfg = ConfigSet(name=DATA_CFG, config_file=template_file, schema_file=SCHEMA_FILE)
        cfg = ConfigSet(name=APP_CFG, config_file=config_file)

        # write template file to manifest_yaml under LANDING_PATH
        landing_path = cfg.get_value(name=DATA_CFG, jsonpath='LANDING_PATH')
        if not os.path.exists(landing_path):
            os.makedirs(landing_path)
        shutil.copy(template_file, os.path.join(landing_path, "manifest.yaml"))

        # subprocess - create proxy table
        if 'delta' in processes or 'all' in processes:
            exit_code = create_proxy_table(config_file)
            if exit_code != 0:
                logger.error("Delta table creation had errors. Exiting.")
                return


def create_proxy_table(config_file):

    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="data_processing.pathology.proxy_table.generate")

    # setup for using external py in udf
    sys.path.append("data_processing/common")
    importlib.import_module("data_processing.common.EnsureByteContext")
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    # use spark to read data from file system and write to parquet format_type
    logger.info("generating binary proxy table... ")

    dicom_path = os.path.join(cfg.get_value(name=DATA_CFG, jsonpath='LANDING_PATH'), const.DICOM_TABLE)


    with CodeTimer(logger, 'load dicom files'):
        print (cfg.get_value(name=DATA_CFG, jsonpath='SOURCE_PATH') + "**" + cfg.get_value(name=DATA_CFG, jsonpath='FILE_TYPE'))
        for path in glob.glob(cfg.get_value(name=DATA_CFG, jsonpath='SOURCE_PATH') + "**" + cfg.get_value(name=DATA_CFG, jsonpath='FILE_TYPE'), recursive=True):
            print (path)

        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
        generate_uuid_udf = udf(generate_uuid, StringType())
        parse_openslide_udf = udf(parse_openslide, MapType(StringType(), StringType()))

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*."+cfg.get_value(name=DATA_CFG, jsonpath='FILE_TYPE')). \
            option("recursiveFileLookup", "true"). \
            load(cfg.get_value(name=DATA_CFG, jsonpath='SOURCE_PATH')). \
            drop("content").\
            withColumn("wsi_record_uuid", lit(generate_uuid_udf  (col("path")))).\
            withColumn("metadata",        lit(parse_openslide_udf(col("path"))))

        df.printSchema()
        df.show()


    if TRY_S3==True:
        upload_to_s3_udf = udf (upload_to_s3, StringType())
        bucket = cfg.get_value(name=DATA_CFG, jsonpath='PROJECT').lower()
        minioClient = Minio(
            'dlliskimind1.mskcc.org:9001',
            access_key='mind',
            secret_key='mskcc123',
            secure=False,
        )
    
        if not minioClient.bucket_exists(bucket):
            minioClient.make_bucket(bucket)
        df.withColumn("bucket", lit(bucket)).withColumn("result", upload_to_s3_udf(col("path"), col("bucket"))).show()
    # parse all dicoms and save

    processed_count = df.count()
    logger.info("Processed {} whole slide images out of total {} files".format(processed_count,cfg.get_value(name=DATA_CFG, jsonpath='FILE_TYPE_COUNT')))
    return exit_code



if __name__ == "__main__":
    cli()
