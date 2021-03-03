from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource

from data_processing.common.custom_logger import init_logger
from data_processing.common.Node import Node
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const
from data_processing.common.config import ConfigSet

import os, shutil, sys, importlib, json, yaml, subprocess, time, uuid, requests
import pandas as pd

app    = Flask(__name__)
api = Api(app, version='1.1', title='cohortManager', description='Manages and exposes study cohort and associated data', ordered=True)

logger = init_logger("flask-mind-server.log")
cfg    = ConfigSet(name=const.APP_CFG,  config_file="config.yaml")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

HOSTNAME     =     cfg.get_value("APP_CFG::cohortManager_host")
PORT         = int(cfg.get_value("APP_CFG::cohortManager_port"))

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ============================================================================================
# get-put-cohort
@api.route('/mind/api/v1/cohort/<cohort_id>', 
    methods=['PUT', 'GET'],
    doc={"description": "Create a cohort and get information (the patient listing) from a cohort"}
)
@api.doc(
    params={'cohort_id': 'Cohort identifier, must be unique if creating a new cohort'},
    responses={200:"Success", 201:"Created successfully", 400:"Invalid", 404:"Cohort not found"}
)
class createOrGetCohort(Resource):
    def put(self, cohort_id):
        """ Create new cohort """
        if ":" in cohort_id: return make_response("Invalid cohort name, only use alphanumeric characters", 400)

        cohort = Node("cohort", cohort_id)
        create_res = conn.query(f""" CREATE (co:{cohort.get_create_str()}) RETURN co""")
        match_res  = conn.query(f""" MATCH  (co:{cohort.get_match_str()} ) RETURN co""")

        if not create_res is None: 
            return make_response("Created successfully", 201)
        elif not match_res is None:           
            return make_response("Cohort already exists", 200)
        else:
            return make_response("Something went wrong, maybe bad query", 400)


    def get(self, cohort_id):
            """ Retrieve listing for cohort """

            cohort = Node("cohort", cohort_id)
            co_res = conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ )
            px_res = conn.query(f""" MATCH (co:{cohort.get_match_str()})-[:INCLUDE]-(px:patient) RETURN px """ )

            if co_res is None:
                return make_response("Bad query", 400)
            if len(co_res)==0:
                return make_response("Cohort not found, please create it first", 400)
        
            # Build summary responses:
            cohort_summary = co_res[0].data()['co']
            cohort_summary['Patients'] = []
            for rec in px_res:
                px_dict     = rec.data()['px']
                patient_id  = px_dict['name']
                px_dict['Patient Accessions'] = requests.get(f'http://{HOST}:5004/mind/api/v1/patient/{cohort_id}/{patient_id}').json()
                cohort_summary['Patients'].append(px_dict)
            return jsonify(cohort_summary)
           
# --------------------------------------------------------------------------------------------


# ============================================================================================
# get-put-cohort
@api.route('/mind/api/v1/cohort/<cohort_id>/<patient_id>', 
    methods=['PUT', 'DELETE'],
    doc={"description": "Modify/Update the patient listing within a cohort"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient identifier to modify'}
)
class modifyPatientInCohort(Resource):
    # Add (include) patient, inverse of removePatient
    def put(self, cohort_id, patient_id):
        """ (Re)-include patient with cohort"""
        cohort  = Node("cohort", cohort_id)
        patient = Node("patient", patient_id, properties={"namespace":cohort_id})

        res = conn.query(f"""MATCH (co:{cohort.get_match_str()}) MATCH (px:{patient.get_match_str()}) MERGE (co)-[r:INCLUDE]-(px) RETURN r""")
        return ("Added {} patients to cohort".format(len(res)))

    # Remove (exclude) patient, inverse of addPatient
    def delete(self, cohort_id, patient_id):
        """ Exclude patient from cohort"""
        cohort  = Node("cohort", cohort_id)
        patient = Node("patient", patient_id, properties={"namespace":cohort_id})
        print ((f"""MATCH (co:{cohort.get_match_str()})-[r:INCLUDE]-(px:{patient.get_match_str()}) DELETE r RETURN r"""))

        res = conn.query(f"""MATCH (co:{cohort.get_match_str()})-[r:INCLUDE]-(px:{patient.get_match_str()}) DELETE r RETURN r""")
        return ("Deleted {} patients from cohort".format(len(res)))
# --------------------------------------------------------------------------------------------


# ============================================================================================
@api.route('/mind/api/v1/container/<cohort_id>/<container_type>/<container_id>', 
    methods=['PUT'],
    doc={"description": "Create a container"}
)
@api.doc(params={'cohort_id': 'Cohort Identifier', 'container_type': 'Type in [scan, patient, slide]', 'container_id':'Unique container identifier'})
class createContainer(Resource):

    def put(self, cohort_id, container_type,  container_id):
            """ Create new container """
            cohort    = Node("cohort", cohort_id)
            container = Node(container_type, container_id)
            container.set_namespace(cohort_id)

            if ":" in container_id: 
                return make_response("Invalid patient name, only use alphanumeric characters", 400)

            print (f""" MATCH (co:{cohort.get_match_str()}) RETURN co """)
            if not len(conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ ))==1: 
                return make_response("No cohort namespace found", 300)

            create_res = conn.query(f""" CREATE (container:{container.get_create_str()}) RETURN container""")
            match_res  = conn.query(f"""
                MATCH (px:{container.get_create_str()})
                MATCH (co:{cohort.get_match_str()})
                MERGE (co)-[r:INCLUDE]-(px)
                RETURN px
                """
            )
            if not create_res is None: 
                return make_response("Created successfully", 201)
            elif not match_res is None:           
                return make_response(f"Container at already exists at {container.get_match_str()}", 200)
            else:
                return make_response("Bad query", 400)
# --------------------------------------------------------------------------------------------
# ============================================================================================
@api.route('/mind/api/v1/patient/<cohort_id>/<patient_id>', 
    methods=['GET', 'PUT'],
    doc={"description": "Create a patient and get information about that patient"}
)
@api.doc(params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient Identifier, must be unique if creating a new patient'})
class createOrGetPatient(Resource):
    def get(self, cohort_id, patient_id):
        """ Retrieve case listing for patient"""
        
        # Matches (cohort <include> patients <has_case> cases)
        patient = Node("patient", patient_id, properties={"namespace":cohort_id})
        res = conn.query(f""" MATCH (px:{patient.get_match_str()})-[:HAS_CASE]-(cases:accession) RETURN cases """)

        all_case = []
        for rec in res:
            case_dict = rec.data()['cases']
            all_case.append(case_dict)
        return jsonify(all_case)

    def put(self, cohort_id, patient_id):
            """ Create new patient """
            cohort  = Node("cohort", cohort_id)
            patient = Node("patient", patient_id, properties={"namespace":cohort_id})

            if ":" in patient_id: 
                return make_response("Invalid patient name, only use alphanumeric characters", 400)

            if not len(conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ ))==1: 
                return make_response("No cohort namespace found", 300)

            create_res = conn.query(f""" CREATE (px:{patient.get_create_str()}) RETURN px """)
            match_res  = conn.query(f"""
                MATCH (px:{patient.get_create_str()})
                MATCH (co:{cohort.get_match_str()})
                MERGE (co)-[r:INCLUDE]-(px)
                RETURN px
                """
            )
            if not create_res is None: 
                return make_response("Created successfully", 201)
            elif not match_res is None:           
                return make_response("Patient already exists", 200)
            else:
                return make_response("Bad query", 400)
# --------------------------------------------------------------------------------------------


# ============================================================================================
@api.route('/mind/api/v1/patient/<cohort_id>/<patient_id>/<case_list>', 
    methods=['PUT', 'DELETE', 'GET'],
    doc={"description": "Modify/Update the case listing of a patients"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'patient_id': 'Patient Identifier', 'case_list':'Comma seperated list of accession numbers to add/remove'},
    responses={200:"Success", 400:"Failed to add patient"}
)
class addOrRemoveCases(Resource):
    def get(self, cohort_id, patient_id, case_list):
        """ Get container listings for a given case list """
        cohort = Node("cohort", cohort_id)
        if not len(conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        patient = Node("patient", patient_id, properties={"namespace":cohort_id})
        res = conn.query(f"""
            MATCH (px:{patient.get_match_str()})
            -[:HAS_CASE]->(cases:accession) 
            -[:HAS_SCAN]->(sc:scan) 
            -[:HAS_DATA]->(data) 
            WHERE cases.AccessionNumber IN [{case_list}] 
            RETURN sc.SeriesInstanceUID, data
            """
        )

        if res is None: 
            return make_response (f"Bad query", 400)
        else:
            collection = {}
            for rec in res: collection[rec.data()['sc.SeriesInstanceUID']] = []
            for rec in res: collection[rec.data()['sc.SeriesInstanceUID']].append(rec.data()['data'])

            return jsonify (collection)

    def put(self, cohort_id, patient_id, case_list):
        """ Add case listing to patient """
        cohort = Node("cohort", cohort_id)
        if not len(conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        patient = Node("patient", patient_id, properties={"namespace":cohort_id})
        res = conn.query(f"""
            MATCH (px:{patient.get_match_str()})
            MATCH (cases:accession) 
            WHERE cases.AccessionNumber IN [{case_list}] 
            MERGE (px)-[r:HAS_CASE]->(cases) 
            RETURN px, cases, r
            """
        )

        if res is None: 
            return make_response (f"Bad query", 400)
        else:
            dict_res = [rec.data()['cases'] for rec in res]
            return make_response (f"Added {patient_id} with {len(dict_res)} cases: {dict_res}", 200)

    def delete(self, cohort_id, patient_id, case_list):
        """ Remove case listing from patient """
        cohort = Node("cohort",  properties={"CohortID":cohort_id})
        if not len(conn.query(f""" MATCH (co:{cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        patient = Node("patient", patient_id, properties={"namespace":cohort_id})
        res = conn.query(f"""
            MATCH (px:{patient.get_match_str()})-[r:HAS_CASE]->(cases:accession)
            WHERE cases.AccessionNumber IN [{case_list}] 
            DELETE r RETURN r
            """
        )
        if res is None: 
            return make_response (f"No matching cases found for {patient_id}", 400)
        else:
            return ("Deleted {} cases from cohort".format(len(res)))
# --------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(host=HOSTNAME,port=PORT, threaded=True, debug=False)

