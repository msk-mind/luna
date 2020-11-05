'''
Created on October 29, 2020

@author: pashaa@mskcc.org
'''

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType
from data_processing.common.EnsureByteContext import EnsureByteContext 

import pydicom
import time
from io import BytesIO
import os, shutil
import json
import yaml, os
import subprocess
from filehash import FileHash
from distutils.util import strtobool

logger = init_logger()


def generate_uuid(path, content):

    file_path = path.split(':')[-1]
    content = BytesIO(content)

    with EnsureByteContext():
        dcm_hash = FileHash('sha256').hash_file(content)

    dicom_record_uuid = f'DICOM-{dcm_hash}'
    return dicom_record_uuid


def parse_dicom_from_delta_record(record):
    
    dirs, filename  = os.path.split(record.path)

    dataset = pydicom.dcmread(BytesIO(record.content))

    kv = {}
    types = set()

    for elem in dataset.iterall():
        types.add(type(elem.value))
        if type(elem.value) in [int, float, str]: 
            kv[elem.keyword] = str(elem.value)
        if type(elem.value) in [pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS, pydicom.valuerep.PersonName, pydicom.uid.UID]: 
            kv[elem.keyword] = str(elem.value)
        if type(elem.value) in [list, pydicom.multival.MultiValue]:
            kv[elem.keyword] = "//".join([str(x) for x in elem.value])
        # not sure how to handle a sequence!
        # if type(elem.value) in [pydicom.sequence.Sequence]: print ( elem.keyword, type(elem.value), elem.value)

    kv['dicom_record_uuid'] = record.dicom_record_uuid 

    with open("jsons/"+filename, 'w') as f:
        json.dump(kv, f)



@click.command()
@click.option('-t', '--template_file', default=None, type=click.Path(exists=True),
              help="path to yaml template file containing information required for radiology proxy data ingestion. "
                   "See data_processing/radiology/proxy_table/data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
@click.option('-s', '--skip_transfer',
              help='do not transfer files from xnat mount, use existing files located at (by default): /dicom')
def cli(template_file, config_file, skip_transfer):
    """
    This module generates a set of proxy tables for radiology data based on information specified in the tempalte file.

    Example:
        python -m data_processing.radiology.proxy_table.generate \
        --template_file {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
        --skip_transfer true
        
    """
    logger = init_logger()
    logger.info('data_ingestions_template: ' + template_file)
    logger.info('config_file: ' + config_file)
    logger.info('skip transfer: ' + skip_transfer)
    
    start_time = time.time()
    
    # setup env variables
    setup_environment_from_yaml(template_file)

    # write template file to manifest_yaml under dest_dir
    if not os.path.exists(os.environ["dataset_name"]):
        os.makedirs(os.environ["dataset_name"])        
    shutil.copy(template_file, os.environ["dataset_name"])

    # subprocess call will preserve environmental variables set by the parent thread.
    if not bool(strtobool(skip_transfer)):
        transfer_files()

    # subprocess - create proxy table
    if not os.path.exists("jsons"):
        os.makedirs("jsons")

    create_proxy_table(config_file)
    print("--- Finished building proxy table in %s seconds ---" % (time.time() - start_time))

def setup_environment_from_yaml(template_file):
     # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)
    
    print(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var])

def transfer_files():
    start_time = time.time()
    transfer_cmd = ["time", "./data_processing/radiology/proxy_table/transfer_files.sh"]
    
    try:
        exit_code = subprocess.call(transfer_cmd)
        print("--- Finished transfering files in %s seconds ---" % (time.time() - start_time))
    except Exception as err:
        logger.error(("Error Transfering files with rsync" + str(err)))
        return 
        
    if exit_code != 0:
        logger.error(("Error Transfering files - Non-zero exit code: " + str(exit_code)))
    
    return 


def create_proxy_table(config_file):

    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    # use spark to read data from file system and write to parquet format_type
    logger.info("generating binary proxy table... ")

    dicom_path = os.path.join(os.environ["dataset_name"], "table", "dicom") 
    dicom_header_path = os.path.join(os.environ["dataset_name"], "table", "dicom_header")

    with CodeTimer(logger, 'delta table create'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load(os.environ["destination_path"])

        generate_uuid_udf = udf(generate_uuid, StringType())
        df = df.withColumn("dicom_record_uuid", lit(generate_uuid_udf(df.path, df.content)))

        df.coalesce(128).write.format(os.environ["format_type"]) \
            .mode("overwrite") \
            .save(dicom_path)
    
    # parse all dicom files
    with CodeTimer(logger, 'read and parse dicom'):
        df.foreach(parse_dicom_from_delta_record)

    # save parsed json headers to tables
    header = spark.read.json("jsons")
    header.write.format(os.environ["format_type"]) \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .save(dicom_header_path)

    if os.path.exists("jsons"):
        shutil.rmtree("jsons")

    processed_count = df.count()
    logger.info("Processed {} dicom headers out of total {} dicom files".format(processed_count, os.environ["file_count"]))
    assert processed_count == int(os.environ["file_count"])
    df = spark.read.format(os.environ["format_type"]).load(dicom_header_path)
    df.printSchema()
    df.show(2, False)


if __name__ == "__main__":
    cli()
