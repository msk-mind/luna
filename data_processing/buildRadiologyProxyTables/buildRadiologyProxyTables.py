
"""
To start a server: python3 -m data_processing.buildRadiologyProxyTables.py (Recommended on sparky1)
"""

from flask import Flask, request

from common.CodeTimer import CodeTimer
from common.custom_logger import init_logger
from common.sparksession import SparkConfig
from common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
# spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")

# ==================================================================================================
# Service functions
# ==================================================================================================

def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info("Setting up environment:")
    logger.info(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var]).strip()

def teardown_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info("Tearing down enviornment")

    # delete all fields from template as env variables
    for var in template_dict:
        del os.environ[var]

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


def create_proxy_table(config_file):

    exit_code = 0
    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")

    # setup for using external py in udf
    sys.path.append("data_processing/common")
    importlib.import_module("data_processing.common.EnsureByteContext")
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    # use spark to read data from file system and write to parquet format_type
    logger.info("generating binary proxy table... ")

    dicom_path = os.path.join(os.environ["LANDING_PATH"], const.DICOM_TABLE)

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

    dicom_path = os.path.join(os.environ["LANDING_PATH"], const.DICOM_TABLE)

    # Which properties to include in dataset node
    dataset_ext_properties = [
        'LANDING_PATH',
        'DATASET_NAME',
        'REQUESTOR',
        'REQUESTOR_DEPARTMENT',
        'REQUESTOR_EMAIL',
        'PROJECT',
        'SOURCE',
        'MODALITY',
    ]

    dataset_props = list ( set(dataset_ext_properties).intersection(set(os.environ.keys())))

    prop_string = ','.join(['''{0}: "{1}"'''.format(prop, os.environ[prop]) for prop in dataset_props])
    conn.query(f'''MERGE (n:dataset{{{prop_string}}})''')

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        df_dcmdata = spark.read.format("delta").load(dicom_path)

        tuple_to_add = df_dcmdata.select("metadata.PatientName", "metadata.SeriesInstanceUID")\
            .groupBy("PatientName", "SeriesInstanceUID")\
            .count()\
            .toPandas()

    with CodeTimer(logger, 'syncronize graph'):

        for index, row in tuple_to_add.iterrows():
            query ='''MATCH (das:dataset {{DATASET_NAME: "{0}"}}) MERGE (px:xnat_patient_id {{value: "{1}"}}) MERGE (sc:scan {{SeriesInstanceUID: "{2}"}}) MERGE (px)-[r1:HAS_SCAN]->(sc) MERGE (das)-[r2:HAS_PX]-(px)'''.format(os.environ['DATASET_NAME'], row['PatientName'], row['SeriesInstanceUID'])
            logger.info (query)
            conn.query(query)
    logger.info (f"Dataset {os.environ['DATASET_NAME']} added successfully!")


# ==================================================================================================
# Routes
# ==================================================================================================
"""
Example request:
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TEMPLATE":"path/to/template.yaml"}' \
  http://pllimsksparky1:5000/mind/api/v1/buildRadiologyProxyTables
"""
@app.route('/mind/api/v1/buildRadiologyProxyTables', methods=['POST'])
def buildRadiologyProxyTables():
    data = request.json
    if not "TEMPLATE" in data.keys(): return "You must supply a template file."

    logger.setLevel('INFO')
    logger.info("Radiology buildRadiologyProxyTables enter.....")
    setup_environment_from_yaml(data["TEMPLATE"])
    create_proxy_table(os.environ['SPARK_CONFIG'])
    teardown_environment_from_yaml(data["TEMPLATE"])
    logger.info("Radiology buildRadiologyProxyTables exit.")

    return "Radiology proxy delta tables creation is successful"

"""
Example request:
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TEMPLATE":"path/to/template.yaml"}' \
  http://pllimsksparky1:5000/mind/api/v1/buildRadiologyGraph
"""
@app.route('/mind/api/v1/buildRadiologyGraph', methods=['POST'])
def buildRadiologyGraph():
    data = request.json
    if not "TEMPLATE" in data.keys(): return "You must supply a template file."

    logger.setLevel('INFO')
    logger.info("Radiology buildRadiologyGraph enter.....")
    setup_environment_from_yaml(data["TEMPLATE"])
    add_scan_to_graph(os.environ['SPARK_CONFIG'])
    teardown_environment_from_yaml(data["TEMPLATE"])
    logger.info("Radiology buildRadiologyGraph exit.")

    return "Radiology buildRadiologyGraph successful"


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'], debug=True)
