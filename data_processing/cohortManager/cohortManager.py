from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource
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
api = Api(app, version='1.1', title='cohortManager', description='Manages and exposes study cohort and associated data', ordered=True)

logger = init_logger("flask-mind-server.log")
cfg    = ConfigSet(name=const.APP_CFG,  config_file="config.yaml")
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
@api.route('/mind/api/v1/patient/list/<cohort_id>', 
    methods=['GET'],
    doc={"description": "List all the patients and recursively, their cases, scans, and available data"}
)
@api.doc(params={'cohort_id': 'Cohort Identifier'})
class listPatients(Resource):
    def get(self, cohort_id):
        """ Retrieve patient listing """

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
            px_dict['cases'] = requests.get(f'http://{HOST}:5004/mind/api/v1/case/list/{cohort_id}/{patient_id}').json()
            all_px.append(px_dict)
        return jsonify(all_px)

# List all the cases
@api.route('/mind/api/v1/case/list/<cohort_id>/<patient_id>', 
    methods=['GET'],
    doc={"description": "List all cases for a given patient and recursively their scans and available data"}
)
@api.doc(params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient Identifier'})
class listCases(Resource):
    def get(self, cohort_id, patient_id):
        """ Retrieve case listing """

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
            case_dict['scans'] = requests.get(f'http://{HOST}:5004/mind/api/v1/scan/list/{cohort_id}/{patient_id}/{accession_number}').json()
            all_case.append(case_dict)
        return jsonify(all_case)

# List all the scans
@api.route('/mind/api/v1/scan/list/<cohort_id>/<patient_id>/<accession_number>', 
    methods=['GET'],
    doc={"description": "List all scans for a given case accession number and recursively their available data"}
)
@api.doc(params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient Identifier', 'accession_number':'Case Accession Number'})
class listScans(Resource):
    def get (self, cohort_id, patient_id, accession_number):
        """ Retrieve case listing """

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
            scan_dict['data'] = requests.get(f'http://{HOST}:5004/mind/api/v1/container/scan/{scan_id}').json()
            all_scan.append(scan_dict)
        return jsonify(all_scan)

# List all the data
@api.route('/mind/api/v1/container/<container_type>/<container_id>', 
    methods=['GET'],
    doc={"description": "List all available data for a given a scan container"}
)
@api.doc(params={'container_type': 'Container type, e.g. scan, slide', 'container_id': 'Unique container identifier'})
class listContainer(Resource):
    def get (self, container_type, container_id):
        """ Retrieve container data listing """

        # Matches (scan <has_data> data)
        res = conn.query(f"""
            MATCH (sc:{container_type})-[:HAS_DATA]-(data) 
            WHERE id(sc)={container_id} 
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
@api.route('/mind/api/v1/patient/new/<cohort_id>/<patient_id>/<case_list>', 
    methods=['POST'],
    doc={"description": "Creates a new patient initalized with some cases"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'patient_id': 'New patient identifier to create', 'case_list':'Comma seperated list of accession numbers to add'},
    responses={200:"Success", 500:"Failed to add patient"}
)
class newPatient(Resource):
    def post(self, cohort_id, patient_id, case_list):
        """ Create new patient """

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
        if len (res)==0: 
            return make_response("No cases", 500)
        else:
            return make_response (f"Added {patient_id} to {cohort_id} with {len(res)} cases: {res}", 200)

@api.route('/mind/api/v1/patient/<cohort_id>/<patient_id>', 
    methods=['PUT', 'DELETE'],
    doc={"description": "Modify the status of a patient within a cohort"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient identifier to modify'}
)
class updatePatient(Resource):
    # Add (include) patient, inverse of removePatient
    def put(self, cohort_id, patient_id):
        """ (Re)-include patient """

        res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}}) MATCH (px:patient{{PatientID:'{patient_id}'}}) MERGE (co)-[r:INCLUDE]-(px) RETURN r""")
        return ("Added {} patients from cohort".format(len(res)))

    # Remove (exclude) patient, inverse of addPatient
    def delete(self, cohort_id, patient_id):
        """ Exclude patient """

        res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[r:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}}) DELETE r RETURN r""")
        return ("Deleted {} patients from cohort".format(len(res)))

if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5004, threaded=True, debug=False)

