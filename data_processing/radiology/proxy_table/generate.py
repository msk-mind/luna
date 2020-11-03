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
import yaml, os
import subprocess

ENVIRONMENTAL_VARS = ['host', 'source_path', 'destination_path',  'files_count']

def parse_dicom_from_delta_record(record):
    dataset = dcmread(BytesIO(record.content))
    print(dataset)
    exit()

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
    # TODO: fix, pass actual booleans through makefile instead of true/false
    if skip_transfer.lower() == "true":
        # subprocess - transfer files if needed
        transfer_cmd = "time ./data_processing/radiology/proxy_table/transfer_files.sh $({0})".format(template_file)
        subprocess.call(transfer_cmd, shell=True)
        print("--- Finished transfering files in %s seconds ---" % (time.time() - start_time))

        # reset timer if transferred files to get accurate proxy building time
        start_time = time.time()

    # subprocess - create proxy table
    create_proxy_table(spark, logger, "dicom")
    print("--- Finished building proxy table in %s seconds ---" % (time.time() - start_time))

def create_proxy_table(spark, logger, dest_dir):
    accum = spark.sparkContext.accumulator(0)

    # use spark to read data from file system and write to parquet format
    logger.info("generating binary proxy table... ")

    # dest_dir = "dicom"
    with CodeTimer(logger, 'delta table create'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load("/gpfs/mskmindhdp_emc/data/radiology/BC_16-512_MR_20201028_UWpYXajT5F")

        df.coalesce(128).write.format(format) \
            .mode("overwrite") \
            .save(dest_dir)

    # use spark to read data from delta table into memory
    with CodeTimer(logger, 'delta table load'):
        header = df.foreach(parse_dicom_from_delta_record)
    
    df.write.format(format).mode("overwrite").save("dicom_header")

    # parse all dicom files
    with CodeTimer(logger, 'read and parse dicom'):
        df.foreach(parse_dicom_from_delta_record)

    logger.info("number of dcms processed: " + str(accum.value))


if __name__ == "__main__":
    cli()