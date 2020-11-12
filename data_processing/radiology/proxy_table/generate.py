'''
Created on October 29, 2020

@author: pashaa@mskcc.org
'''

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, MapType

import pydicom
import time
from io import BytesIO
import os, shutil, sys, importlib
import json
import yaml, os
import subprocess
from filehash import FileHash

logger = init_logger()

def generate_uuid(path, content):

    file_path = path.split(':')[-1]
    content = BytesIO(content)

    import EnsureByteContext 
    with EnsureByteContext.EnsureByteContext():
        dcm_hash = FileHash('sha256').hash_file(content)

    dicom_record_uuid = f'DICOM-{dcm_hash}'
    return dicom_record_uuid


def parse_dicom_from_delta_record(path, content):
    
    dirs, filename  = os.path.split(path)

    dataset = pydicom.dcmread(BytesIO(content))

    kv = {}
    types = set()
    skipped_keys = []

    for elem in dataset.iterall():
        types.add(type(elem.value))
        if type(elem.value) in [int, float, str]: 
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS, pydicom.valuerep.PersonName, pydicom.uid.UID]: 
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [list, pydicom.multival.MultiValue]:
            kv[elem.keyword] = "//".join([str(x) for x in elem.value])
        else:
            skipped_keys.append(elem.keyword)
        # not sure how to handle a sequence!
        # if type(elem.value) in [pydicom.sequence.Sequence]: print ( elem.keyword, type(elem.value), elem.value)
    
    if "" in kv:
        kv.pop("")
    return kv


@click.command()
@click.option('-t', '--template_file', default=None, type=click.Path(exists=True),
              help="path to yaml template file containing information required for radiology proxy data ingestion. "
                   "See data_processing/radiology/proxy_table/data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
@click.option('-p', '--process_string', default='all',
              help='comma separated list of processes to run or replay: e.g. transfer,delta,graph, or all')
def cli(template_file, config_file, process_string):
    """
    This module generates a set of proxy tables for radiology data based on information specified in the tempalte file.

    Example:
        python -m data_processing.radiology.proxy_table.generate \
        --template_file {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
        --process_string transfer,delta 
        
    """
    processes = process_string.lower().strip().split(",")
    logger.info('data_ingestions_template: ' + template_file)
    logger.info('config_file: ' + config_file)
    logger.info('processes: ' + str(processes))
   
    start_time = time.time()
    
    # setup env variables
    setup_environment_from_yaml(template_file)

    # write template file to manifest_yaml under LANDING_PATH
    if not os.path.exists(os.environ["LANDING_PATH"]):
        os.makedirs(os.environ["LANDING_PATH"])
    shutil.copy(template_file, os.path.join(os.environ["LANDING_PATH"], "manifest.yaml"))

    # subprocess call will preserve environmental variables set by the parent thread.
    if 'transfer' in processes or 'all' in processes:
        exit_code = transfer_files()
        if exit_code != 0:
            return

    # subprocess - create proxy table
    if 'delta' in processes or 'all' in processes:
        exit_code = create_proxy_table(config_file)
        if exit_code != 0:
            logger.error("Delta table creation had errors. Exiting.")
            return

    # update graph
    if 'graph' in processes or 'all' in processes:
        update_graph(config_file)

    # subprocess - sync to graph
    logger.info("--- Finished radiology proxy etl in %s seconds ---" % (time.time() - start_time))

def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)
    
    logger.info(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var]).strip()

def transfer_files():
    start_time = time.time()
    transfer_cmd = ["time", "./data_processing/radiology/proxy_table/transfer_files.sh"]
    
    try:
        exit_code = subprocess.call(transfer_cmd)
        logger.info("--- Finished transfering files in %s seconds ---" % (time.time() - start_time))
    except Exception as err:
        logger.error(("Error Transfering files with rsync" + str(err)))
        return -1 
        
    if exit_code != 0:
        logger.error(("Error Transfering files - Non-zero exit code: " + str(exit_code)))
    
    return exit_code


def create_proxy_table(config_file):
    
    exit_code = 0
    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    # setup for using external py in udf
    sys.path.append("data_processing/common")
    importlib.import_module("data_processing.common.EnsureByteContext")
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    # use spark to read data from file system and write to parquet format_type
    logger.info("generating binary proxy table... ")

    dicom_path = os.path.join(os.environ["TABLE_PATH"], "dicom") 
    
    with CodeTimer(logger, 'load dicom files'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load(os.environ["RAW_DATA_PATH"])

        generate_uuid_udf = udf(generate_uuid, StringType())
        df = df.withColumn("dicom_record_uuid", lit(generate_uuid_udf(df.path, df.content)))

    # parse all dicoms and save
    with CodeTimer(logger, 'parse and save dicom'):
        parse_dicom_from_delta_record_udf = udf(parse_dicom_from_delta_record, MapType(StringType(), StringType()))
        header = df.withColumn("metadata", lit(parse_dicom_from_delta_record_udf(df.path, df.content)))
    
        header.coalesce(6144).write.format(os.environ["FORMAT_TYPE"]) \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(dicom_path)

    processed_count = header.count()
    logger.info("Processed {} dicom headers out of total {} dicom files".format(processed_count, os.environ["FILE_TYPE_COUNT"]))

    # validate and show created dataset
    if processed_count != int(os.environ["FILE_TYPE_COUNT"]):
        exit_code = 1
    df = spark.read.format(os.environ["FORMAT_TYPE"]).load(dicom_path)
    df.printSchema()
    return exit_code

def update_graph(config_file):
    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    # Open a connection to the ID graph database
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    dicom_path = os.path.join(os.environ["TABLE_PATH"], "dicom")

    # Which properties to include in dataset node
    dataset_ext_properties = [
        'REQUESTOR',
        'REQUESTOR_DEPARTMENT',
        'REQUESTOR_EMAIL',
        'PROJECT',
        'SOURCE',
        'MODALITY',
    ]
     
    dataset_props = list ( set(dataset_ext_properties).intersection(set(os.environ.keys())))  
    dataset_props.insert(0, 'TABLE_PATH')
    dataset_props.insert(0, 'DATASET_NAME')
    
    prop_string = ','.join(['''{0}: "{1}"'''.format(prop, os.environ[prop]) for prop in dataset_props])
    conn.query(f'''MERGE (n:dataset{{{prop_string}}})''')

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        df_dcmdata = spark.read.format("delta").load(dicom_path)
    
        tuple_to_add = df_dcmdata.select("PatientName", "SeriesInstanceUID")\
            .groupBy("PatientName", "SeriesInstanceUID")\
            .count()\
            .toPandas()
    
    with CodeTimer(logger, 'syncronize graph'):
    
        for index, row in tuple_to_add.iterrows():
            query ='''MATCH (das:dataset {{DATASET_NAME: "{0}"}}) MERGE (px:xnat_patient_id {{value: "{1}"}}) MERGE (sc:scan {{SeriesInstanceUID: "{2}"}}) MERGE (px)-[r1:HAS_SCAN]->(sc) MERGE (das)-[r2:HAS_PX]-(px)'''.format(os.environ['DATASET_NAME'], row['PatientName'], row['SeriesInstanceUID'])
            logger.info (query)
            conn.query(query)


if __name__ == "__main__":
    cli()
