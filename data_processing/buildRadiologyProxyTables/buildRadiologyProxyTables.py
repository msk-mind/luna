#!/gpfs/mskmindhdp_emc/sw/env/bin/python3

"""
To start a server: ./data_processing_app.py (Recommended on sparky1)
"""

from flask import Flask, request
import os
import sys

sys.path.append(os.path.abspath('../'))

from common.CodeTimer import CodeTimer
from common.custom_logger import init_logger
from common.sparksession import SparkConfig
from common.Neo4jConnection import Neo4jConnection
import importlib


from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType

import pydicom
import time
from io import BytesIO
import os, shutil, sys, importlib
import json
import yaml, os
import subprocess
from filehash import FileHash
from distutils.util import strtobool

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
# spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")


@app.route('/')
def index():
    # setup env variables
    return "Hello, MSK!"

"""
Build Radiology proxy delta tables
"""


@app.route('/mind/api/v1/buildRadiologyProxyTables', methods=['GET', 'POST'])
def buildRadiologyProxyTables():
    logger.setLevel('INFO')
    logger.info("Radiology proxy enter.....")
    setup_environment_from_yaml(os.environ['PATH_TO_TEMPLATE_FILE'])
    create_proxy_table(os.environ['SPARK_CONFIG'])
    logger.info("Radiology proxy delta tables creation is successful")

    return "Radiology proxy delta tables creation is successful"


def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var]).strip()


def generate_uuid(path, content):
    file_path = path.split(':')[-1]
    content = BytesIO(content)

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        dcm_hash = FileHash('sha256').hash_file(content)

    dicom_record_uuid = f'DICOM-{dcm_hash}'
    return dicom_record_uuid


def parse_dicom_from_delta_record(record, json_path):
    dirs, filename = os.path.split(record.path)

    dataset = pydicom.dcmread(BytesIO(record.content))

    kv = {}
    types = set()
    skipped_keys = []

    for elem in dataset.iterall():
        types.add(type(elem.value))
        if type(elem.value) in [int, float, str]:
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS,
                                  pydicom.valuerep.PersonName, pydicom.uid.UID]:
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [list, pydicom.multival.MultiValue]:
            kv[elem.keyword] = "//".join([str(x) for x in elem.value])
        else:
            skipped_keys.append(elem.keyword)
        # not sure how to handle a sequence!
        # if type(elem.value) in [pydicom.sequence.Sequence]: print ( elem.keyword, type(elem.value), elem.value)

    kv['dicom_record_uuid'] = record.dicom_record_uuid
    with open(os.path.join(json_path, filename), 'w') as f:
        json.dump(kv, f)


def create_proxy_table(config_file):
    logger.info("create_proxy_table enter.....")
    spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")
    json_path = os.environ["TMPJSON_PATH"]

    # setup for using external py in udf
    # sys.path.append("data_processing/common")
    data_directory = r"common"
    os.environ['PATH'] += os.path.pathsep + data_directory

    importlib.import_module("common.EnsureByteContext")
    spark.sparkContext.addPyFile("../data_processing/common/EnsureByteContext.py")
    # use spark to read data from file system and write to parquet format_type
    logger.info("generating binary proxy table... ")

    dicom_path = os.path.join(os.environ["TABLE_PATH"], "dicom")
    dicom_header_path = os.path.join(os.environ["TABLE_PATH"], "dicom_header")

    with CodeTimer(logger, 'delta table create'):
        spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

        df = spark.read.format("binaryFile"). \
            option("pathGlobFilter", "*.dcm"). \
            option("recursiveFileLookup", "true"). \
            load(os.environ["RAW_DATA_PATH"])

        generate_uuid_udf = udf(generate_uuid, StringType())
        df = df.withColumn("dicom_record_uuid", lit(generate_uuid_udf(df.path, df.content)))

        df.coalesce(512).write.format(os.environ["FORMAT_TYPE"]) \
            .mode("overwrite") \
            .save(dicom_path)

    # parse all dicom files
    with CodeTimer(logger, 'read and parse dicom'):
        df.foreach(lambda x: parse_dicom_from_delta_record(x, json_path))

    # save parsed json headers to tables
    with CodeTimer(logger, 'read jsons and save dicom headers'):
        header = spark.read.json(json_path)
        header.coalesce(256).write.format(os.environ["FORMAT_TYPE"]) \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .save(dicom_header_path)

    # clean up temporary jsons
    if os.path.exists(json_path):
        shutil.rmtree(json_path)

    processed_count = df.count()
    logger.info(
        "Processed {} dicom headers out of total {} dicom files".format(processed_count, os.environ["FILE_COUNT"]))

    # validate and show created dataset
    assert processed_count == int(os.environ["FILE_COUNT"])
    df = spark.read.format(os.environ["FORMAT_TYPE"]).load(dicom_header_path)
    df.printSchema()


"""
Example request:
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TABLE_PATH":"/gpfs/mskmindhdp_emc/user/aukermaa/radiology/TEST_16-158_CT_20201028/table", "DATASET_NAME":"API_TESTING"}' \
  http://pllimsksparky1:5000/mind/api/v1/graph
"""


@app.route('/mind/api/v1/buildRadiologyGraph', methods=['POST'])
def buildRadiologyGraph():
    data = request.json

    # Open a connection to the ID graph database
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    dicom_header_path = os.path.join(data["TABLE_PATH"], "dicom_header")

    prop_string = ','.join(['''{0}: "{1}"'''.format(prop, data[prop]) for prop in data.keys()])
    conn.query(f'''MERGE (n:dataset{{{prop_string}}})''')

    with CodeTimer(logger, 'setup proxy table'):
        # Reading dicom and opdata
        try:
            df_dcmdata = spark.read.format("delta").load(dicom_header_path)
        except:
            return (f"{dicom_header_path} either does not exist or is not a delta table")

        tuple_to_add = df_dcmdata.select("PatientName", "SeriesInstanceUID") \
            .groupBy("PatientName", "SeriesInstanceUID") \
            .count() \
            .toPandas()

    with CodeTimer(logger, 'syncronize graph'):

        for index, row in tuple_to_add.iterrows():
            query = '''MATCH (das:dataset {{DATASET_NAME: "{0}"}}) MERGE (px:xnat_patient_id {{value: "{1}"}}) MERGE (sc:scan {{SeriesInstanceUID: "{2}"}}) MERGE (px)-[r1:HAS_SCAN]->(sc) MERGE (das)-[r2:HAS_PX]-(px)'''.format(
                data['DATASET_NAME'], row['PatientName'], row['SeriesInstanceUID'])
            logger.info(query)
            conn.query(query)
    return (f"Dataset {data['DATASET_NAME']} added successfully!")


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'], debug=True)
