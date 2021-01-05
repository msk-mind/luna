from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.utils import secure_filename

from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const
from data_processing.common.config import ConfigSet
from pyspark.sql.types import StringType, IntegerType, StructType, StructField

import os, shutil, sys, importlib, json, yaml, subprocess, time, uuid, requests
import pandas as pd

import threading

app    = Flask(__name__)
logger = init_logger("flask-mind-server.log")
cfg    = ConfigSet(name=const.APP_CFG,  config_file="config.yaml")
spark  = SparkConfig().spark_session(const.APP_CFG, "data_processing.radiology.api.5004")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
lock   = threading.Lock()

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

STREAMS = {}
METHODS = {}
HOST = os.environ['HOSTNAME']

# ==================================================================================================
# Routes to list things
# ==================================================================================================

# List all the patients
@app.route('/mind/api/v1/listPatients/<cohort_id>', methods=['GET'])
def listPatients(cohort_id):

    # Matches (cohort <include> patients)
    res = conn.query(f"""
        MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[:INCLUDE]-(px:patient) 
        RETURN px
        """
    )

    all_px = []
    for rec in res:
        px_dict = rec.data()['px']
        patient_id = px_dict['PatientID']
        px_dict['cases'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listCases/{cohort_id}/{patient_id}').json()
        all_px.append(px_dict)
    return jsonify(all_px)

# List all the cases
@app.route('/mind/api/v1/listCases/<cohort_id>/<patient_id>', methods=['GET'])
def listCases(cohort_id, patient_id):

    # Matches (cohort <include> patients <has_case> cases)
    res = conn.query(f"""
        MATCH (co:cohort{{CohortID:'{cohort_id}'}})
        -[:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}})
        -[:HAS_CASE]-(cases:accession) 
        RETURN cases
        """
    )

    all_case = []
    for rec in res:
        case_dict = rec.data()['cases']
        accession_number   = case_dict['AccessionNumber']
        case_dict['scans'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listScans/{cohort_id}/{patient_id}/{accession_number}').json()
        all_case.append(case_dict)
    return jsonify(all_case)

# List all the scans
@app.route('/mind/api/v1/listScans/<cohort_id>/<patient_id>/<accession_number>', methods=['GET'])
def listScans(cohort_id, patient_id, accession_number):
    
    # Matches (cohort <include> patients <has_case> cases <has_scan> scan <include> cohort)
    res = conn.query(f"""
        MATCH (co:cohort{{CohortID:'{cohort_id}'}})
        -[:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}})
        -[:HAS_CASE]-(cases:accession{{AccessionNumber:'{accession_number}'}})
        -[:HAS_SCAN]-(sc:scan)
        -[:INCLUDE]-(co)
        RETURN sc, id(sc)
        """
    )

    all_scan = []
    for rec in res:
        scan_dict = rec.data()['sc']
        scan_id   = rec.data()['id(sc)']
        scan_dict['id']   = scan_id
        scan_dict['data'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listData/{scan_id}').json()
        all_scan.append(scan_dict)
    return jsonify(all_scan)

# List all the data
@app.route('/mind/api/v1/listData/<id>', methods=['GET'])
def listData(id):

    # Matches (scan <has_data> data)
    res = conn.query(f"""
        MATCH (sc:scan)-[:HAS_DATA]-(data) 
        WHERE id(sc)={id} 
        RETURN data, labels(data)
        """
    )

    all_data = []
    for rec in res:
        data_dict = rec.data()['data']
        data_dict['type'] = rec.data()['labels(data)'][0]
        all_data.append(data_dict)
    return jsonify(all_data)



# ==================================================================================================
# Routes to add manage cohort/study structure (new/add/remove patient)
# ==================================================================================================

# Add a new (and include) patient with cases
@app.route('/mind/api/v1/newPatient/<cohort_id>/<patient_id>/<case_list>', methods=['GET'])
def newPatient(cohort_id, patient_id, case_list):

    res = conn.query(f"""
        MATCH (cases:accession) 
        WHERE cases.AccessionNumber IN [{case_list}] 
        MERGE (px:patient{{PatientID:'{patient_id}'}})
        MERGE (co:cohort{{CohortID:'{cohort_id}'}})
        MERGE (co)-[r1:INCLUDE]-(px)
        MERGE (co)-[r2:INCLUDE]-(cases)
        MERGE (px)-[r3:HAS_CASE]->(cases) 
        RETURN cases
        """
    )
    res = [rec.data()['cases'] for rec in res]
    
    return (f"Added {patient_id} to {cohort_id} with {len(res)} cases: {res}")

# Add (include) patient, inverse of removePatient
@app.route('/mind/api/v1/addPatient/<cohort_id>/<patient_id>', methods=['GET'])
def addPatient(cohort_id, patient_id):
    res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}}) MATCH (px:patient{{PatientID:'{patient_id}'}}) MERGE (co)-[r:INCLUDE]-(px) RETURN r""")
    return ("Added {} patients from cohort".format(len(res)))

# Remove (exclude) patient, inverse of addPatient
@app.route('/mind/api/v1/removePatient/<cohort_id>/<patient_id>', methods=['GET'])
def removePatient(cohort_id, patient_id):
    res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[r:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}}) DELETE r RETURN r""")
    return ("Deleted {} patients from cohort".format(len(res)))



# ==================================================================================================
# Routes to extract scan data
# ==================================================================================================
# Initilize scans with DCM data
@app.route('/mind/api/v1/initScans/<cohort_id>/<query>', methods=['GET'])
def initScans(cohort_id, query):
    # Get relevant patients and cases
    res_tree = conn.query(f"""
        MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[:INCLUDE]-(cases:accession)-[*]->(das:dataset) WHERE das.DATA_TYPE="DCM" \
        RETURN DISTINCT co, cases
        """
    )

    # Get the "Parquet Dataset"
    res_data = conn.query(f"""
        MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[:INCLUDE]-(cases:accession)-[*]->(das:dataset) WHERE das.DATA_TYPE="DCM" \
        RETURN DISTINCT das
        """
    )

    logger.info("Length of dataset = {}".format(len(res_data)))

    cSchema = StructType([StructField("CohortID", StringType(), True), StructField("AccessionNumber", StringType(), True)])
    df_cohort = spark.createDataFrame(([(x.data()['co']['CohortID'], x.data()['cases']['AccessionNumber']) for x in res_tree]),schema=cSchema)

    if not len(res_data)==1: return make_response(("Operations only support singleton datasets right now", 500))

    df = spark.read\
        .format("delta")\
        .load(res_data[0]['das']['TABLE_LOCATION'])\
        .select("path", \
            "AccessionNumber", \
            "SeriesInstanceUID", \
            "metadata.SeriesNumber", \
            "metadata.SeriesDescription", \
            "metadata.PerformedProcedureStepDescription", \
            "metadata.ImageType", \
            "metadata.InstanceNumber")\
        .where(query)\
        .where("InstanceNumber='1'")\
        .orderBy("AccessionNumber") \
        .join(df_cohort, ['AccessionNumber'])

    for index, row in df.toPandas().iterrows():
        prop_string = ",".join( ['''{0}:"{1}"'''.format(key, row[key]) for key in row.index] )
        query = """
                MATCH (co:cohort{{CohortID:'{2}'}})
                MATCH (sc:scan{{SeriesInstanceUID:'{0}'}})
                MERGE (data:dcm{{{1}}})
                MERGE (sc)-[:HAS_DATA]-(data)
                MERGE (co)-[:INCLUDE]-(sc) RETURN data
                """.format(row['SeriesInstanceUID'], prop_string, cohort_id)
        logger.info (query)
        conn.query(query)

    return make_response("Done", 200)


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5004, threaded=True, debug=True)

