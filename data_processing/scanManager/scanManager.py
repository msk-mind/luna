from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, fields

from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.GraphEnum import Node
from data_processing.common.config import ConfigSet
from pyspark.sql.types import StringType, IntegerType, StructType, StructField

import os, shutil, sys, importlib, json, yaml, subprocess, time, uuid, requests
import pandas as pd
import subprocess
import threading
from multiprocessing import Pool
from filehash import FileHash



app    = Flask(__name__)
api = Api(app, version='1.1', title='scanManager', description='Manages and exposes study scans and associated data', ordered=True)

logger = init_logger("flask-mind-server.log")
cfg    = ConfigSet("APP_CFG",  config_file="config.yaml")
spark  = SparkConfig().spark_session("APP_CFG", "data_processing.radiology.api.5003")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
lock   = threading.Lock()

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

STREAMS = {}
METHODS = {}
HOST = os.environ['HOSTNAME']

model = api.model("Post Method",
    { 
        "image":    fields.String(description="Transform function name", required=True, example="data_processing.scanManager.generateScan"),
        "file_ext": fields.String(description="File extension to save volumentric image", required=True, example="mhd")
    }
)
@api.route('/mind/api/v1/scans/<cohort_id>', methods=['GET'])
class getScansCohort(Resource):
    def get(self, cohort_id):
        "Return list of scan container IDs for a given cohort"
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (co:{n_cohort.match()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:INCLUDE]-(co) \
            RETURN DISTINCT id(sc)
            """
        )
        return jsonify([rec.data()['id(sc)'] for rec in res])

@api.route('/mind/api/v1/scans/<cohort_id>/<case_id>', methods=['GET'])
class getScansCases(Resource):
    def get(self, cohort_id, case_id):
        "Return list of scan container IDs within a case for a given cohort"
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (co:{n_cohort.match()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:INCLUDE]-(co) \
            WHERE cases.AccessionNumber="{case_id}"
            RETURN DISTINCT id(sc)
            """
        )
        return jsonify([rec.data()['id(sc)'] for rec in res])


@api.route('/mind/api/v1/container/<cohort_id>/<container_id>', methods=['GET','POST'])
class manageContainer(Resource):
    def get(self, cohort_id, container_id):
        "Ping container"
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (sc:scan)-[:HAS_DATA]-(data)
            WHERE id(sc)={container_id}
            RETURN data
            """
        )
        return jsonify([rec.data()['data'] for rec in res])
    def post(self, cohort_id, container_id):
        "Add a record to a container"
        
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        properties = request.json

        if not "path" in properties.keys():
            return make_response("You must supply a path!", 400)

        if not "type" in properties.keys():
            return make_response("You must supply a type!", 400)  

        if not os.path.exists(properties['path']):
            return make_response("File does not exist!", 400)

        properties['Namespace'] = cohort_id
        properties['RecordID']  = "mha" + "-" + str(FileHash('sha256').hash_file(properties['path']))

        try:
            n_data = Node(properties["type"],  properties=properties)
        except:
            return make_response("Failed to configure node, bad type???", 401)

        conn.query(f""" 
            MATCH (sc:scan) WHERE id(sc)={container_id}
            MERGE (da:{n_data.create()})
            MERGE (sc)-[:HAS_DATA]->(da)"""
        )
        return make_response(f"Added {n_data.match()} to container {container_id}", 200)



@api.route('/mind/api/v1/methods/<cohort_id>/<method_id>/run', 
    methods=['POST'],
    doc={"description": "Add new or view current namespace methods"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'method_id': 'Method name'},
    responses={200:"Success", 400: "Method already exists"}
)
class runMethods(Resource):
    def post(self, cohort_id, method_id):
        """ Run a method """
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['Namespace'] = cohort_id
        properties['MethodID']  = method_id
        n_method = Node("method",  properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.match()}) RETURN me""")

        if not len(res)==1:
            return make_response("No method namespace found", 300)

        method_config = res[0].data()['me']
        scan_ids = requests.get(f'http://{HOST}:5003/mind/api/v1/scans/{cohort_id}').json()

        logger.info(method_config)
        logger.info(scan_ids)

        args_list = []
        for scan_id in scan_ids:
            args_list.append(["python3","-m",method_config["image"],"-c", cohort_id, "-s", str(scan_id), "-m", method_id])

        p = Pool(25)
        p.map(subprocess.call, args_list)

        return "Done"


@api.route('/mind/api/v1/methods/<cohort_id>/<method_id>', 
    methods=['POST','GET', 'DELETE'],
    doc={"description": "Add new or view current namespace methods"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'method_id': 'Method name'},
    responses={200:"Success", 400: "Method already exists"}
)
class methods(Resource):
    @api.expect(model)
    def post(self, cohort_id, method_id):
        """ Create new method configuration """
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}

        properties['Namespace'] = cohort_id
        properties['MethodID']  = method_id
        properties['image']     = request.json["image"]
        n_method = Node("method",  properties=properties)

        res = conn.query(f"""CREATE (me:{n_method.create()}) RETURN me""")
        if res is None: return make_response(f"Method at {cohort_id}::{method_id} already exists!", 400)
        
        with open(f'{method_id}.json', 'w') as outfile:
            json.dump(request.json, outfile, indent=4)

        return make_response(f"Created new method at {cohort_id}::{method_id}!", 200)

    def get(self, cohort_id, method_id):
        """ Get configuration for a method """
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['Namespace'] = cohort_id
        properties['MethodID']  = method_id
        n_method = Node("method",  properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.match()}) RETURN me""")

        if not len(res)==1:
            return make_response("Method not found", 300)
        else:
            return jsonify(res[0].data()['me'])

    def delete(self, cohort_id, method_id):
        """ Delete configuration for a method """
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['Namespace'] = cohort_id
        properties['MethodID']  = method_id
        n_method = Node("method",  properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.match()}) DETACH DELETE me RETURN me""")

        return make_response("Deleted method")


# ============================================================================================
@api.route('/mind/api/v1/init/<cohort_id>/<query>', 
    methods=['PUT'],
    doc={"description": "Initalize scans under a cohort namespace given a query on fields in [SeriesNumber, SeriesDescription, PerformedProcedureStepDescription, ImageType]"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'query': 'Where clause for scan metadata to add'},
    responses={200:"Success"}
)
class initScans(Resource):
    def put(self, cohort_id, query):
        """ Add some scans to the cohort """
        n_cohort = Node("cohort",  properties={"CohortID":cohort_id})

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.match()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res_tree = conn.query(f"""
            MATCH (co:{n_cohort.match()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:HAS_DATA]-(das:dataset) WHERE das.DATA_TYPE="DCM" \
            RETURN DISTINCT co, cases
            """
        )

        # Get the "Parquet Dataset"
        res_data = conn.query(f"""
            MATCH (co:{n_cohort.match()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:HAS_DATA]-(das:dataset) WHERE das.DATA_TYPE="DCM" \
            RETURN DISTINCT das
            """
        )

        logger.info("Length of dataset = {}".format(len(res_data)))
        logger.info("Query = {}".format(query))

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

        count = 0
        for index, row in df.toPandas().iterrows():
            count += 1
            properties = dict(row)
            properties['RecordID'] = "DCM-" + properties["SeriesInstanceUID"]
            properties['Namespace'] = cohort_id
            n_meta = Node("dicom",    properties=properties)
            n_scan = Node("scan",     properties={"QualifiedPath":properties["SeriesInstanceUID"]})
            query = f"""
                    MATCH  (sc:{n_scan.match()})
                    MATCH  (co:{n_cohort.match()})
                    CREATE (da:{n_meta.create()})
                    MERGE (sc)-[:HAS_DATA]->(da)
                    MERGE (co)-[:INCLUDE]->(sc) 
                    """
            conn.query(query)

        return make_response(f"Done, added {count} nodes", 200)


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5003, threaded=True, debug=True)
