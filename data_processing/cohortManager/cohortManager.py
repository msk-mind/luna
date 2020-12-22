from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.utils import secure_filename

from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const
from data_processing.common.config import ConfigSet
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from checksumdir import dirhash

import os, shutil, sys, importlib, json, yaml, subprocess, time, uuid, requests
import pandas as pd
import itk

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
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
    res = conn.query(f"""MATCH (px:patient) where px.cohort='{cohort_id}' and px.active=True RETURN px""")
    all_px = []
    for rec in res:
        px_dict = rec.data()['px']
        patient_id = px_dict['name']
        px_dict['cases'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listCases/{cohort_id}/{patient_id}').json()
        all_px.append(px_dict)
    return jsonify(all_px)

# List all the cases
@app.route('/mind/api/v1/listCases/<cohort_id>/<patient_id>', methods=['GET'])
def listCases(cohort_id, patient_id):
    res = conn.query(f"""MATCH (px:patient)-[:PROXY]-(proxy)-[:HAS_CASE]-(case:case) where px.name='{patient_id}' and px.cohort='{cohort_id}' and case.cohort='{cohort_id}' RETURN case""")
    all_case = []
    for rec in res:
        case_dict = rec.data()['case']
        case_id   = case_dict['AccessionNumber']
        case_dict['scans'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listScans/{cohort_id}/{patient_id}/{case_id}').json()
        all_case.append(case_dict)
    return jsonify(all_case)

# List all the scans
@app.route('/mind/api/v1/listScans/<cohort_id>/<patient_id>/<case_id>', methods=['GET'])
def listScans(cohort_id, patient_id, case_id):
    res = conn.query(f"""MATCH (px:patient)-[:PROXY]-(proxy)-[:HAS_CASE]-(case:case)-[:HAS_SCAN]-(scan:scan) where px.name='{patient_id}' and px.cohort='{cohort_id}' and case.cohort='{cohort_id}' and case.AccessionNumber='{case_id}' RETURN scan, id(scan)""")
    all_scan = []
    for rec in res:
        scan_dict = rec.data()['scan']
        scan_id   = rec.data()['id(scan)']
        scan_dict['id']   = scan_id
        scan_dict['data'] = requests.get(f'http://{HOST}:5004/mind/api/v1/listData/{scan_id}').json()
        all_scan.append(scan_dict)
    return jsonify(all_scan)

# List all the scans
@app.route('/mind/api/v1/listData/<id>', methods=['GET'])
def listData(id):
    res = conn.query(f"""MATCH (scan:scan)-[:HAS_DATA]-(data) WHERE id(scan)={id} RETURN data, labels(data)""")
    all_data = []
    for rec in res:
        data_dict = rec.data()['data']
        data_dict['type'] = rec.data()['labels(data)'][0]
        all_data.append(data_dict)
    print (all_data)
    return jsonify(all_data)



# ==================================================================================================
# Routes to add manage cohort/study structure (add/remove patient, add cases, add scans)
# ==================================================================================================

# Add (xnat) patient
@app.route('/mind/api/v1/addXnatPatient/<cohort_id>/<patient_id>', methods=['GET'])
def addPatient(cohort_id, patient_id):
    res = conn.query(f"""MATCH (n) WHERE n.PatientID='{patient_id}' RETURN n""")
    if len(res) == 0:
        return ("No patients found")
    elif len(res) == 1:
        conn.query(f"""MATCH (n) WHERE n.PatientID='{patient_id}' MERGE (m:patient{{name:'{patient_id}', cohort:'{cohort_id}'}}) SET m.active=True  MERGE (n)-[:PROXY]-(m)""")
        return (f"Added {patient_id} to {cohort_id}")
    else:
        return ("Malformed patient id returned multiple types")

# Remove (deactive) patient
@app.route('/mind/api/v1/removePatient/<cohort_id>/<patient_id>', methods=['GET'])
def removePatient(cohort_id, patient_id):
    res = conn.query(f"""MATCH (m:patient{{name:'{patient_id}', cohort:'{cohort_id}'}}) SET m.active=False RETURN m""")
    return ("Deleted {} patients from cohort".format(len(res)))

# Add proxy cases
@app.route('/mind/api/v1/initCases/<cohort_id>', methods=['GET'])
def initCases(cohort_id):
    res = conn.query(f"""
        MATCH
            (px:patient{{cohort:'{cohort_id}'}})
            -[:PROXY]-(px_proxy)
            -[:HAS_CASE]-(case_proxy:case)
        WHERE
            case_proxy.type='radiology'
        MERGE (case_new:case{{ AccessionNumber : case_proxy.AccessionNumber + '-{cohort_id}' }})
            SET case_new.cohort='{cohort_id}'
            SET case_new.active=True
        MERGE (case_proxy)-[:PROXY]-(case_new)
        MERGE (px_proxy)-[:HAS_CASE]-(case_new)
        RETURN case_new
        """
    )
    if res is None: res=[]
    return "Total {} new cases".format(len(res))

# Add scans
@app.route('/mind/api/v1/initScans/<cohort_id>/<query>', methods=['GET'])
def initScans(cohort_id, query):
    # Look for DCM type scans in proxy case
    res_tree = conn.query(f"""
        MATCH (case_proxy:case{{cohort:'{cohort_id}'}})-[:PROXY]-(case:case)-[*]->(das:dataset) WHERE das.DATA_TYPE="DCM" \
        RETURN DISTINCT case_proxy, case
        """
    )
    res_data = conn.query(f"""
        MATCH (case_proxy:case{{cohort:'{cohort_id}'}})-[:PROXY]-(case:case)-[*]->(das:dataset) WHERE das.DATA_TYPE="DCM" \
        RETURN DISTINCT das
        """
    )
    logger.info("Length of dataset = {}".format(len(res_data)))

    cSchema = StructType([StructField("ProxyAccessionNumber", StringType(), True), StructField("AccessionNumber", StringType(), True)])
    df_cohort = spark.createDataFrame(([(x.data()['case_proxy']['AccessionNumber'], x.data()['case']['AccessionNumber']) for x in res_tree]),schema=cSchema)

    if not len(res_data)==1: make_response(("Operations only support singleton datasets", 500))

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

    df.show()

    for index, row in df.toPandas().iterrows():
        prop_string = ",".join( ['''{0}:"{1}"'''.format(key, row[key]) for key in row.index] )
        query = '''
                MATCH (case_proxy:case{{AccessionNumber:'{0}'}})
                MERGE (sc:scan{{{1}}}) SET sc.cohort='{2}'
                MERGE (case_proxy)-[:HAS_SCAN]-(sc) RETURN case_proxy
                '''.format(row['ProxyAccessionNumber'], prop_string, cohort_id)
        logger.info (query)
        conn.query(query)

    return jsonify([rec.data() for rec in res_tree])


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5004, threaded=True, debug=False)
