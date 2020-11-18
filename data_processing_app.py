#!/gpfs/mskmindhdp_emc/sw/env/bin/python3

"""
To start a server: ./data_processing_app.py (Recommended on sparky1)
"""

from flask import Flask, request, jsonify
import os
import pydicom
import time
from io import BytesIO
import os, shutil, sys, importlib
import json
import yaml
import subprocess
import random 
from filehash import FileHash

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")

def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)
    
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


"""
curl http://pllimsksparky1:5000/
"""
@app.route('/')
def index():
    return "Hello, MSK!"

"""
curl http://pllimsksparky1:5000/mind/api/v1/env
"""
@app.route('/mind/api/v1/env')
def env():
    return ",".join([key + '=' + os.environ[key] for key in os.environ.keys()])

"""
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TEMPLATE":"/home/aukermaa/dev_b/data-processing/ingestion.yaml"}' \
  http://pllimsksparky1:5000/mind/api/v1/transfer
"""
@app.route('/mind/api/v1/transfer', methods=['POST'])
def transfer():
    config = request.json
    setup_environment_from_yaml(config["TEMPLATE"])


    teardown_environment_from_yaml(config["TEMPLATE"])
    return str(config) 

"""
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TEMPLATE":"/home/aukermaa/dev_b/data-processing/ingestion.yaml"}' \
  http://pllimsksparky1:5000/mind/api/v1/transfer
"""
@app.route('/mind/api/v1/delta', methods=['POST'])
def delta():
    config = request.json
    setup_environment_from_yaml(config["TEMPLATE"])


    teardown_environment_from_yaml(config["TEMPLATE"])
    return str(config) 

"""
curl http://pllimsksparky1:5000/mind/api/v1/datasets
"""
@app.route('/mind/api/v1/datasets', methods=['GET'])
def datasets_list():
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    res = conn.query(f'''MATCH (n:dataset) RETURN n''')
    return jsonify([rec.data()['n']['DATASET_NAME'] for rec in res])

"""
curl http://pllimsksparky1:5000/mind/api/v1/datasets/MY_DATASET
"""
@app.route('/mind/api/v1/datasets/<name>', methods=['GET'])
def datasets_get(name):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    res = conn.query(f'''MATCH (n:dataset) WHERE n.DATASET_NAME = '{name}' RETURN n''')
    return jsonify([rec.data()['n'] for rec in res])

"""
curl http://pllimsksparky1:5000/mind/api/v1/datasets/MY_DATASET
"""
@app.route('/mind/api/v1/datasets/<name>/countDicom', methods=['GET'])
def datasets_countDicom(name):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    res = conn.query(f'''MATCH (n:dataset) WHERE n.DATASET_NAME = '{name}' RETURN n''')
    das = [rec.data()['n'] for rec in res]
    if len (das) < 1: return f"Sorry, I cannot find that dataset {name}"
    if len (das) > 1: return f"Sorry, this dataset has multiplicity > 1"
    count = spark.read.format("delta").load(das[0]["TABLE_PATH"] + "/dicom").count()
    das[0][f"countDicom"] = count
    return jsonify(das) 
"""
Example request:
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TABLE_PATH":"/gpfs/mskmindhdp_emc/user/aukermaa/radiology/TEST_16-158_CT_20201028/table", "DATASET_NAME":"API_TESTING"}' \
  http://pllimsksparky1:5000/mind/api/v1/graph
"""
@app.route('/mind/api/v1/graph', methods=['POST'])
def graph():
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

        tuple_to_add = df_dcmdata.select("PatientName", "SeriesInstanceUID")\
            .groupBy("PatientName", "SeriesInstanceUID")\
            .count()\
            .toPandas()

    with CodeTimer(logger, 'syncronize graph'):

        for index, row in tuple_to_add.iterrows():
            query ='''MATCH (das:dataset {{DATASET_NAME: "{0}"}}) MERGE (px:xnat_patient_id {{value: "{1}"}}) MERGE (sc:scan {{SeriesInstanceUID: "{2}"}}) MERGE (px)-[r1:HAS_SCAN]->(sc) MERGE (das)-[r2:HAS_PX]-(px)'''.format(data['DATASET_NAME'], row['PatientName'], row['SeriesInstanceUID'])
            logger.info (query)
            conn.query(query)
    return (f"Dataset {data['DATASET_NAME']} added successfully!")

if __name__ == '__main__':
    app.run(host = os.environ['HOSTNAME'], debug=True)
