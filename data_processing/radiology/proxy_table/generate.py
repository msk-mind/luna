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

def parse_dicom_from_delta_record(record):
    dataset = dcmread(BytesIO(record.content))
    if len(dataset.keys()) > 10:
        accum.add(1)

@click.command()
@click.option('-t', '--template', default=None,
              help="path to yaml template file containing information required for radiology proxy data ingestion. "
                   "See data_processing/radiology/proxy_table/data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml',
              help="path to config file containing application configuration. See config.yaml.template")
def cli(template,
        config_file):
    """
    This module generates a set of proxy tables for radiology data based on information specified in the tempalte file.

    Example:
        python -m data_processing.radiology.proxy_table.generate \
        --template {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
    """
    logger = init_logger()
    logger.info('data_ingestions_template: ' + template)
    logger.info('config_file: ' + config_file)

    # Setup Spark context
    import time
    start_time = time.time()


    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    accum = spark.sparkContext.accumulator(0)

    # use spark to read data from file system and write to parquet format
    logger.info("generating binary proxy table... ")
    dest_dir = "dicom"
    with CodeTimer(logger, 'delta table create'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load("mskmind_XNAT")

        df.coalesce(128).write.format(format) \
            .mode("overwrite") \
            .save(dest_dir)

    # use spark to read data from delta table into memory
    with CodeTimer('delta table load', logger):
        df = spark.read.format(format).load(dest_dir)

    # parse all dicom files
    with CodeTimer('read and parse dicom', logger):
        df.foreach(parse_dicom_from_delta_record)

    logger.info("number of dcms processed: " + str(accum.value))


    print("--- Finished in %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    cli()