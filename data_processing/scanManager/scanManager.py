from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, fields

from data_processing.common.custom_logger    import init_logger
from data_processing.common.sparksession     import SparkConfig
from data_processing.common.Neo4jConnection  import Neo4jConnection
from data_processing.common.Node      import Node
from data_processing.common.config    import ConfigSet
from data_processing.common.Container import Container

from pyspark.sql.types import StringType, IntegerType, StructType, StructField

import os, shutil, sys, json, subprocess, uuid, requests
import pandas as pd

from multiprocessing    import Pool
from concurrent.futures import ProcessPoolExecutor

from data_processing.scanManager.extractRadiomics import extract_radiomics_with_container
from data_processing.scanManager.windowDicoms     import window_dicom_with_container
from data_processing.scanManager.generateScan     import enerate_scan_with_container

"""
Required config:
Environmental:
    - HOSTNAME - Hostname of running process
    - GRAPH_URI - URI for graph database
Config.yaml:
    - scanManager_port      - What port to publish API
    - scanManager_processes - Number of concurrent jobs
"""
container_params = {
        'GRAPH_URI':  os.environ['GRAPH_URI'],
        'GRAPH_USER': "neo4j",
        'GRAPH_PASSWORD': "password"
    }

# Setup configurations
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")
VERSION      = "branch:"+subprocess.check_output(["git","rev-parse" ,"--abbrev-ref", "HEAD"]).decode('ascii').strip()
HOSTNAME     = os.environ["HOSTNAME"]
PORT         = int(cfg.get_value("APP_CFG::scanManager_port"))
NUM_PROCS = int(cfg.get_value("APP_CFG::scanManager_processes"))

# Setup App/Api
app = Flask(__name__)
api = Api(app, version=VERSION, title='scanManager', description='Manages and exposes study scans and associated data', ordered=True)

# Initialize some important classes
logger = init_logger("flask-mind-server.log")
spark  = SparkConfig().spark_session("APP_CFG", "data_processing.radiology.api.5003")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

# Method param model
model = api.model("method",
    { 
        "image":    fields.String(description="Transform function name",   required=True, example="data_processing.scanManager.generateScan"),
        "params":   fields.String(description="Json formatted parameters", required=True, example={"param1":"some_value"})
    }
)

@api.route('/mind/api/v1/scans/<cohort_id>', methods=['GET'])
class getScansCohort(Resource):
    def get(self, cohort_id):
        """Return list of scan container IDs for a given cohort"""

        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (co:{n_cohort.get_match_str()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:INCLUDE]-(co) \
            RETURN DISTINCT id(sc)
            """
        )
        return jsonify([rec.data()['id(sc)'] for rec in res])

@api.route('/mind/api/v1/scans/<cohort_id>/<case_id>', methods=['GET'])
class getScansCases(Resource):
    def get(self, cohort_id, case_id):
        """Return list of scan container IDs within a case for a given cohort"""

        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (co:{n_cohort.get_match_str()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:INCLUDE]-(co) \
            WHERE cases.AccessionNumber="{case_id}"
            RETURN DISTINCT id(sc)
            """
        )
        return jsonify([rec.data()['id(sc)'] for rec in res])


@api.route('/mind/api/v1/container/<cohort_id>/<container_id>', methods=['GET','POST'])
class manageContainer(Resource):
    def get(self, cohort_id, container_id):

        """ Ping container data """
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1:
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res = conn.query(f"""
            MATCH (container)-[:HAS_DATA]-(data)
            WHERE id(container)={container_id} AND data.namespace='{cohort_id}'
            RETURN data
            """
        )
        return jsonify([rec.data()['data'] for rec in res])
    def post(self, cohort_id, container_id):
        """ Add a record to a container """
        properties = request.json
        
        try:
            n_data = Node(properties['type'], properties['name'], properties=properties)
        except:
            return make_response("Failed to configure node, bad type???", 401)

        container = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(container_id)
        container.add(n_data)
        container.saveAll()

     
        return make_response(f"Added {n_data.get_match_str()} to container {container_id}", 200)


@api.route('/mind/api/v1/methods/<cohort_id>/<method_id>/run', 
    methods=['POST'],
    doc={"description": "Add new or view current namespace methods"}
)
@api.route('/mind/api/v1/methods/<cohort_id>/<method_id>/<container_id>/run', 
    methods=['POST'],
    doc={"description": "Add new or view current namespace methods"}
)
@api.doc(
    params={'cohort_id': 'Cohort Identifier', 'method_id': 'Method name'},
    responses={200:"Success", 400: "Method already exists"}
)
class runMethods(Resource):
    def post(self, cohort_id, method_id, container_id=None):
        """ Run a method """
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['namespace'] = cohort_id
        n_method = Node("method", method_id, properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.get_match_str()}) RETURN me""")

        if not len(res)==1:
            return make_response("No method namespace found", 300)

        method_config = res[0].data()['me']

        a = request.args.get('id', None)

        if container_id==None:
            container_ids = requests.get(f'http://{HOSTNAME}:{PORT}/mind/api/v1/scans/{cohort_id}').json()
        else:
            container_ids = [container_id]

        logger.info(method_config)
        logger.info(container_ids)

        args_list = []

        with ProcessPoolExecutor(NUM_PROCS) as executor:
            if method_config["function"] == 'data_processing.scanManager.windowDicoms':
                for scan_id in container_ids:
                    executor.submit (window_dicom_with_container, cohort_id, str(scan_id), method_id )
            if method_config["function"] == 'data_processing.scanManager.generateScan':
                for scan_id in container_ids:
                    executor.submit (generate_scan_with_container, cohort_id, str(scan_id), method_id )
            if method_config["function"] == 'data_processing.scanManager.extractRadiomics':
                for scan_id in container_ids:
                    executor.submit (extract_radiomics_with_container, cohort_id, str(scan_id), method_id )
            if method_config["function"] == 'data_processing.scanManager.saveRadiomics':
                for scan_id in container_ids:
                    args_list.append(["python3","-m",method_config["function"],"-c", cohort_id, "-s", str(scan_id), "-m", method_id])
                p = Pool(NUM_PROCS)
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
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}

        properties['namespace']    = cohort_id
        properties['function']     = request.json["function"]
        n_method = Node("method", method_id, properties=properties)

        res = conn.query(f"""CREATE (me:{n_method.get_create_str()}) RETURN me""")
        #if res is None: return make_response(f"Method at {cohort_id}::{method_id} already exists!", 400)

        method_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, "methods")
        if not os.path.exists(method_dir): os.makedirs(method_dir)

        with open(os.path.join(method_dir, f'{method_id}.json'), 'w') as outfile:
            json.dump(request.json, outfile, indent=4)

        return make_response(f"Created new method at {cohort_id}::{method_id}!", 200)

    def get(self, cohort_id, method_id):
        """ Get configuration for a method """
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['namespace'] = cohort_id
        n_method = Node("method", method_id, properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.get_match_str()}) RETURN me""")

        if not len(res)==1:
            return make_response("Method not found", 300)
        else:
            return jsonify(res[0].data()['me'])

    def delete(self, cohort_id, method_id):
        """ Delete configuration for a method """
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        properties = {}
        properties['namespace'] = cohort_id
        n_method = Node("method", method_id, properties=properties)

        res = conn.query(f"""MATCH (me:{n_method.get_match_str()}) DETACH DELETE me RETURN me""")

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
        n_cohort = Node("cohort", cohort_id)

        # Check for cohort existence
        if not len(conn.query(f""" MATCH (co:{n_cohort.get_match_str()}) RETURN co """ ))==1: 
            return make_response("No cohort namespace found", 300)

        # Get relevant patients and cases
        res_tree = conn.query(f"""
            MATCH (co:{n_cohort.get_match_str()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:HAS_DATA]-(das:dataset) WHERE das.DATA_TYPE="DCM" \
            RETURN DISTINCT co, cases
            """
        )

        # Get the "Parquet Dataset"
        res_data = conn.query(f"""
            MATCH (co:{n_cohort.get_match_str()})-[:INCLUDE]-(px:patient)-[:HAS_CASE]-(cases:accession)-[:HAS_SCAN]-(sc:scan)-[:HAS_DATA]-(das:dataset) WHERE das.DATA_TYPE="DCM" \
            RETURN DISTINCT das
            """
        )

        logger.info("Length of dataset = {}".format(len(res_data)))
        logger.info("Query = {}".format(query))

        cSchema = StructType([StructField("CohortID", StringType(), True), StructField("AccessionNumber", StringType(), True)])
        df_cohort = spark.createDataFrame(([(x.data()['co']['name'], x.data()['cases']['AccessionNumber']) for x in res_tree]),schema=cSchema)

        df_cohort.show()

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
            properties['path'] = os.path.split(properties['path'])[0]

            n_meta = Node("dicom", 'init-scans', properties=properties)
            
            container = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(row['SeriesInstanceUID'])
            container.add(n_meta)
            container.saveAll()

        return make_response(f"Done, added {count} nodes", 200)


if __name__ == '__main__':
    app.run(host=HOSTNAME,port=PORT, threaded=True, debug=False)
