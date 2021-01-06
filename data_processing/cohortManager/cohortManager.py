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

# # List all the patients
# @api.route('/mind/api/v1/patient/list/<cohort_id>', 
#     methods=['GET'],
#     doc={"description": "List all the patients and recursively, their cases"}
# )
# @api.doc(params={'cohort_id': 'Cohort Identifier'})
# class listPatients(Resource):
#     def get(self, cohort_id):
#         """ Retrieve patient listing """

#         # Matches (cohort <include> patients)
#         res = conn.query(f"""
#             MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[:INCLUDE]-(px:patient) 
#             RETURN px
#             """
#         )

#         all_px = []
#         for rec in res:
#             px_dict = rec.data()['px']
#             patient_id = px_dict['PatientID']
#             px_dict['cases'] = requests.get(f'http://{HOST}:5004/mind/api/v1/case/list/{cohort_id}/{patient_id}').json()
#             all_px.append(px_dict)
#         return jsonify(all_px)


# # List all the scans
# @api.route('/mind/api/v1/scan/list/<cohort_id>/<patient_id>/<accession_number>', 
#     methods=['GET'],
#     doc={"description": "List all scans for a given case accession number and recursively their available data"}
# )
# @api.doc(params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient Identifier', 'accession_number':'Case Accession Number'})
# class listScans(Resource):
#     def get (self, cohort_id, patient_id, accession_number):
#         """ Retrieve scan listing """

#         # Matches (cohort <include> patients <has_case> cases <has_scan> scan <include> cohort)
#         res = conn.query(f"""
#             MATCH (co:cohort{{CohortID:'{cohort_id}'}})
#             -[:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}})
#             -[:HAS_CASE]-(cases:accession{{AccessionNumber:'{accession_number}'}})
#             -[:HAS_SCAN]-(sc:scan)
#             -[:INCLUDE]-(co)
#             RETURN sc, id(sc)
#             """
#         )

#         all_scan = []
#         for rec in res:
#             scan_dict = rec.data()['sc']
#             scan_id   = rec.data()['id(sc)']
#             scan_dict['id']   = scan_id
#             scan_dict['data'] = requests.get(f'http://{HOST}:5004/mind/api/v1/container/scan/{scan_id}').json()
#             all_scan.append(scan_dict)
#         return jsonify(all_scan)

# # List all the data
# @api.route('/mind/api/v1/container/<container_type>/<container_id>', 
#     methods=['GET'],
#     doc={"description": "List all available data for a given a scan container"}
# )
# @api.doc(params={'container_type': 'Container type, e.g. scan, slide', 'container_id': 'Unique container identifier'})
# class listContainer(Resource):
#     def get (self, container_type, container_id):
#         """ Retrieve container data listing """

#         # Matches (scan <has_data> data)
#         res = conn.query(f"""
#             MATCH (sc:{container_type})-[:HAS_DATA]-(data) 
#             WHERE id(sc)={container_id} 
#             RETURN data, labels(data)
#             """
#         )

#         all_data = []
#         for rec in res:
#             data_dict = rec.data()['data']
#             data_dict['type'] = rec.data()['labels(data)'][0]
#             all_data.append(data_dict)
#         return jsonify(all_data)


# ==================================================================================================
# Routes to add manage cohort/study structure (new/add/remove patient)
# ==================================================================================================
# Add a new (and include) patient with cases
@api.route('/mind/api/v1/cohort/<cohort_id>', 
    methods=['PUT', 'GET'],
    doc={"description": "Manage the existence of a cohort"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier'},
    responses={200:"Success", 201:"Created successfully", 400:"Bad query", 404:"Cohort not found"}
)
class cohort(Resource):
    def put(self, cohort_id):
        """ Create new cohort """

        create_res = conn.query(f"""
            CREATE (co:cohort{{CohortID:'{cohort_id}'}})
            RETURN co
            """
        )
        match_res = conn.query(f"""
            MATCH (co:cohort{{CohortID:'{cohort_id}'}})
            RETURN co
            """
        )
        if not create_res is None: 
            return make_response("Created successfully", 201)
        elif not match_res is None:           
            return make_response("Cohort already exists", 200)
        else:
            return make_response("Bad query", 400)


    def get(self, cohort_id):
            """ Retrieve listing for cohort """

            co_res = conn.query(f"""
                MATCH (co:cohort{{CohortID:'{cohort_id}'}})
                RETURN co
                """
            )

            px_res = conn.query(f"""
                MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[:INCLUDE]-(px:patient) 
                RETURN px
                """
            )

            # Build summary responses:
            cohort_summary = co_res[0].data()['co']
            cohort_summary['Patients'] = []
            for rec in px_res:
                px_dict     = rec.data()['px']
                patient_id  = px_dict['PatientID']
                px_dict['Patient Accessions'] = requests.get(f'http://{HOST}:5004/mind/api/v1/patient/{patient_id}').json()
                cohort_summary['Patients'].append(px_dict)
            
            if not co_res is None: 
                return jsonify(cohort_summary)
            else:           
                return make_response("Cohort not found", 404)


# Get patient listing
@api.route('/mind/api/v1/patient/<patient_id>', 
    methods=['GET', 'PUT'],
    doc={"description": "List all cases for a given patient"}
)
@api.doc(params={'patient_id': 'Patient Identifier'})
class listCases(Resource):
    def get(self, patient_id):
        """ Retrieve case listing for patient"""

        # Matches (cohort <include> patients <has_case> cases)
        res = conn.query(f"""
            MATCH (px:patient{{PatientID:'{patient_id}'}})-[:HAS_CASE]-(cases:accession) 
            RETURN cases
            """
        )

        all_case = []
        for rec in res:
            case_dict = rec.data()['cases']
            all_case.append(case_dict)
        return jsonify(all_case)

    def put(self, patient_id):
            """ Create new patient """

            create_res = conn.query(f"""
                CREATE (px:patient{{PatientID:'{patient_id}'}})
                RETURN px
                """
            )
            match_res = conn.query(f"""
                MATCH (px:patient{{PatientID:'{patient_id}'}})
                RETURN px
                """
            )
            if not create_res is None: 
                return make_response("Created successfully", 201)
            elif not match_res is None:           
                return make_response("Patient already exists", 200)
            else:
                return make_response("Bad query", 400)

# Add a new (and include) patient with cases
@api.route('/mind/api/v1/patient/<patient_id>/<case_list>', 
    methods=['PUT', 'DELETE'],
    doc={"description": "Manage the existence of a patients"}
)
@api.doc(
    params={'patient_id': 'New patient identifier to create', 'case_list':'Comma seperated list of accession numbers to add'},
    responses={200:"Success", 400:"Failed to add patient"}
)
class newPatient(Resource):
    def put(self, patient_id, case_list):
        """ Add case listing to patient """

        res = conn.query(f"""
            MATCH (cases:accession) 
            WHERE cases.AccessionNumber IN [{case_list}] 
            MATCH (px:patient{{PatientID:'{patient_id}'}})
            MERGE (px)-[r:HAS_CASE]->(cases) 
            RETURN px, cases, r
            """
        )
        print (res)
        if res is None: 
            return make_response (f"Bad query", 400)
        else:
            dict_res = [rec.data()['cases'] for rec in res]
            return make_response (f"Added {patient_id} with {len(dict_res)} cases: {dict_res}", 200)

    def delete(self, patient_id, case_list):
        """ Remove case listing from patient """

        res = conn.query(f"""
            MATCH (px:patient{{PatientID:'{patient_id}'}})-[r:HAS_CASE]->(cases:accession)
            WHERE cases.AccessionNumber IN [{case_list}] 
            DELETE r RETURN r
            """
        )
        if res is None: 
            return make_response (f"No matching cases found for {patient_id}", 400)
        else:
            return ("Deleted {} cases from cohort".format(len(res)))




@api.route('/mind/api/v1/cohort/<cohort_id>/<patient_id>', 
    methods=['PUT', 'DELETE'],
    doc={"description": "Modify the status of a patient within a cohort"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient identifier to modify'}
)
class updatePatient(Resource):
    # Add (include) patient, inverse of removePatient
    def put(self, cohort_id, patient_id):
        """ (Re)-include patient with cohort"""

        res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}}) MATCH (px:patient{{PatientID:'{patient_id}'}}) MERGE (co)-[r:INCLUDE]-(px) RETURN r""")
        return ("Added {} patients from cohort".format(len(res)))

    # Remove (exclude) patient, inverse of addPatient
    def delete(self, cohort_id, patient_id):
        """ Exclude patient from cohort"""

        res = conn.query(f"""MATCH (co:cohort{{CohortID:'{cohort_id}'}})-[r:INCLUDE]-(px:patient{{PatientID:'{patient_id}'}}) DELETE r RETURN r""")
        return ("Deleted {} patients from cohort".format(len(res)))


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5004, threaded=True, debug=True)

