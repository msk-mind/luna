'''
Created on October 29, 2020

@author: pashaa@mskcc.org
'''
import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from pydicom import dcmread
from io import BytesIO
import os, shutil
import json
import yaml, os
import subprocess
from distutils.util import strtobool

ENVIRONMENTAL_VARS = ['host', 'source_path', 'destination_path',  'files_count']

def parse_dicom_from_delta_record(record):
    dataset = dcmread(BytesIO(record.content))
    kv = {}
    for elem in dataset.iterall():
        kv[elem.keyword] = elem.repval
    
    dirs, filename = os.path.split(record.path)
    with open("jsons/"+filename, 'w') as f:
        print("write " + "jsons/"+filename)
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
        --skip_transfer 
        
    """
    logger = init_logger()
    logger.info('data_ingestions_template: ' + template_file)
    logger.info('config_file: ' + config_file)
    logger.info('skip transfer: ' + skip_transfer)
    # Setup Spark context
    import time
    start_time = time.time()
    format = 'delta'
    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)
    
    print(template_dict)

    # identify fields to set as env variables
    for var in ENVIRONMENTAL_VARS:
        os.environ[var] = str(template_dict[var])

    # subprocess call will preserve environmental variables set by the parent thread.
    if not bool(strtobool(skip_transfer)):
        # subprocess - transfer files if needed
        transfer_cmd = "time ./data_processing/radiology/proxy_table/transfer_files.sh $({0})".format(template_file)
        subprocess.call(transfer_cmd, shell=True)
        print("--- Finished transfering files in %s seconds ---" % (time.time() - start_time))

        # reset timer if transferred files to get accurate proxy building time
        start_time = time.time()

    # subprocess - create proxy table
    if not os.path.exists("jsons"):
        os.makedirs("jsons")

    create_proxy_table(spark, logger, "dicom", format)
    print("--- Finished building proxy table in %s seconds ---" % (time.time() - start_time))

def create_proxy_table(spark, logger, dest_dir, format):

    # use spark to read data from file system and write to parquet format
    logger.info("generating binary proxy table... ")

    # dest_dir = "dicom"
    with CodeTimer(logger, 'delta table create'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load(os.environ["destination_path"])

        df.coalesce(128).write.format(format) \
            .mode("overwrite") \
            .save(dest_dir)
    
    # parse all dicom files
    with CodeTimer(logger, 'read and parse dicom'):
        df.foreach(parse_dicom_from_delta_record)

    # save parsed json headers to tables
    header = spark.read.json("jsons")
    header.write.format(format) \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .save("dicom_header")

    if os.path.exists("jsons"):
        shutil.rmtree("jsons")

    processed_count = df.count()
    print("Processed {} dicom headers out of total {} dicom files".format(processed_count, os.environ["files_count"]))
    df = spark.read.format("delta").load("dicom_header")
    df.printSchema()
    df.show(2, False)


if __name__ == "__main__":
    cli()
