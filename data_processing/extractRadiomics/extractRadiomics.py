from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, fields

from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.GraphEnum import Node
from data_processing.common.config import ConfigSet
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
from checksumdir import dirhash

import os, shutil, sys, importlib, json, yaml, subprocess, time, uuid, requests
import pandas as pd

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import itk

from multiprocessing import Process
import threading

app    = Flask(__name__)
api = Api(app, version='1.1', title='scanManager', description='Manages and exposes study scans and associated data', ordered=True)

logger = init_logger("flask-mind-server.log")
cfg    = ConfigSet("APP_CFG",  config_file="config.yaml")
#spark  = SparkConfig().spark_session("APP_CFG", "data_processing.radiology.api.5003")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
lock   = threading.Lock()

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

STREAMS = {}
METHODS = {}
HOST = os.environ['HOSTNAME']

model = api.model("Post Method",
    { 
        "image":    fields.String(description="Transform function name", required=True, example="generateScan"),
        "file_ext": fields.String(description="File extension to save volumentric image", required=True, example="mhd")
    }
)

# ==================================================================================================
# Starts a spark stream!!!
# ==================================================================================================

# Start stream from stream dir to delta table
def START_STREAM(stream_dir, table_dir):
    ckpt_dir = stream_dir.replace("streams", "checkpointLocation")
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

    logger.info("STARTING STREAM FROM: "+ stream_dir)

    return spark\
    .readStream\
        .option("header", "true")\
        .schema(spark.read.option("header", "true").csv(stream_dir).schema)\
        .csv(stream_dir)\
    .writeStream\
        .format("delta")\
        .option("checkpointLocation", ckpt_dir)\
        .trigger(processingTime="1 minute")\
        .outputMode("append")\
        .start(table_dir)

# ==================================================================================================
# Utility orchestrators
# ==================================================================================================

# Process all scans in a cohort
@app.route('/mind/api/v1/processScans/<cohort_id>/<method_id>', methods=['GET'])
def processScans(cohort_id, method_id):
    ids_ = requests.get(f'http://{HOST}:5003/mind/api/v1/getScanIDs/{cohort_id}').json()

    responses = []
    for id in ids_:
        response = requests.get(f'http://{HOST}:5003/mind/api/v1/radiomics/{method_id}/{id}')
        responses.append(response.text)

    return jsonify(responses)

# Create radiomics extractor
@app.route('/mind/api/v1/configureRadiomics/<cohort_id>/<method_id>', methods=['POST'])
def configureRadiomics(cohort_id, method_id):
    # !!!!!!!!! NOT PROCESS SAFE !!!!!!!!

    with lock:
        if not method_id in METHODS.keys():
            dataset_id = f"RAD_{method_id}"
            METHODS[method_id] = {}
            METHODS[method_id]['config']      = request.json
            METHODS[method_id]['cohort_id']   = cohort_id
            METHODS[method_id]['dataset_id']  = dataset_id
            METHODS[method_id]['streams_dir'] = os.path.join("/gpfs/mskmindhdp_emc/data/", cohort_id, "streams",  dataset_id)
            METHODS[method_id]['dataset_dir'] = os.path.join("/gpfs/mskmindhdp_emc/data/", cohort_id, "tables", dataset_id)
            if not os.path.exists(METHODS[method_id]['streams_dir']): os.makedirs(METHODS[method_id]['streams_dir'])
            if not os.path.exists(METHODS[method_id]['dataset_dir']): os.makedirs(METHODS[method_id]['dataset_dir'])

            print (METHODS)
            return jsonify(METHODS[method_id])

        else:
            return jsonify(METHODS[method_id])


# ==================================================================================================
# Generate/Extract/Call utility functions
# ==================================================================================================

@app.route('/mind/api/v1/generateScan', methods=['POST'])
def generateScan():
    params = request.json

    cohort_id    = params['cohort_id']
    container_id = params['container_id']
    file_ext     = params['file_ext']

    input_nodes = conn.query(f""" MATCH (object:scan)-[:HAS_DATA]-(data:metadata) WHERE id(object)={container_id} and data.Type="dcm" RETURN data""")
    
    if not input_nodes: return "Nothing there!"

    input_data = input_nodes[0].data()['data']

    path        = input_data['path']

    input_dir, filename  = os.path.split(path)
    input_dir = input_dir[input_dir.index("/"):]

    output_dir = os.path.join("/gpfs/mskmindhdp_emc/data/COHORTS", cohort_id, "scans", input_data['SeriesInstanceUID'])
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    PixelType = itk.ctype('signed short')
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(input_dir)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        logger.warning('No DICOMs in: ' + input_dir)
        return make_response("No DICOMs in: " + input_dir, 500)

    logger.info('The directory {} contains {} DICOM Series: '.format(input_dir, str(num_dicoms)))

    n_slices = 0

    for uid in seriesUIDs:
        logger.info('Reading: ' + uid)
        fileNames = namesGenerator.GetFileNames(uid)
        if len(fileNames) < 1: continue

        n_slices = len(fileNames)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()

        outFileName = os.path.join(output_dir, uid + '.' + file_ext)
        writer.SetFileName(outFileName)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())
        logger.info('Writing: ' + outFileName)
        writer.Update()

    properties = {}
    properties['RecordID'] = "SCAN-" + dirhash(input_dir, "sha256")
    properties['Namespace'] = cohort_id
    properties['Type'] = file_ext
    properties['path'] = outFileName
    properties['zdim'] = n_slices

    n_meta = Node("metadata", properties=properties)

    conn.query(f""" 
        MATCH (sc:scan) WHERE id(sc)={container_id}
        MERGE (da:{n_meta.create()})
        MERGE (sc)-[:HAS_DATA]->(da)"""
    )

    return "Done"


# ==================================================================================================
# Initilize scans with DCM data
# ============================================================================================


@api.route('/mind/api/v1/scans/<cohort_id>', methods=['GET'], doc=False)
class getScans(Resource):
    def get(self, cohort_id):
        "Return list of scan container IDs"
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
        return jsonify(  [rec.data()['id(sc)'] for rec in res])

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
        logger.info(method_config)
        scan_ids = requests.get(f'http://{HOST}:5003/mind/api/v1/scans/{cohort_id}').json()
        logger.info(scan_ids)


        procs = []
        for scan_id in scan_ids:
            payload = method_config
            payload['cohort_id'] = cohort_id
            payload['container_id'] = scan_id
            requests.post(f"http://{HOST}:5003/mind/api/v1/generateScan", json=payload)


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

        properties = request.json
        properties['Namespace'] = cohort_id
        properties['MethodID']  = method_id
        n_method = Node("method",  properties=properties)

        res = conn.query(f"""CREATE (me:{n_method.create()}) RETURN me""")
        if res is None: return make_response(f"Method at {cohort_id}::{method_id} already exists!", 400)
        return make_response(f"Created new method at {cohort_id}::{method_id} already exists!", 400)
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
            properties['Type'] = "dcm"
            n_meta = Node("metadata", properties=properties)
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







# # Extract radiomics
# @app.route('/mind/api/v1/radiomics/<method_id>/<id>', methods=['GET'])
# def extractRadiomics(method_id, id):
#     if not method_id in METHODS.keys():
#         return make_response("No extractor configured for " + method_id)

#     scan_node = conn.query(f"""
#         MATCH (object:scan)-[:HAS_DATA]-(image:mhd)
#         MATCH (object:scan)-[:HAS_DATA]-(label:mha)
#         WHERE id(object)={id}
#         RETURN object.cohort, object.AccessionNumber, image.path, label.path"""
#     )

#     if not scan_node:
#         return make_response("Scan is not ready for radiomics (missing annotation?)", 500)

#     scan_node = scan_node[0].data()

#     JOB_CONFIG = METHODS[method_id]

#     config      = JOB_CONFIG['config']
#     streams_dir = JOB_CONFIG['streams_dir']
#     dataset_dir = JOB_CONFIG['dataset_dir']

#     extractor = featureextractor.RadiomicsFeatureExtractor(**config['params'])

#     try:
#         result = extractor.execute(scan_node["image.path"].split(':')[-1], scan_node["label.path"].split(':')[-1])
#     except Exception as e:
#         return make_response(str(e), 200)

#     sers = pd.Series(result)
#     sers["AccessionNumber"] = scan_node["object.AccessionNumber"]
#     sers["config"]          = config
#     sers.to_frame().transpose().to_csv(os.path.join(streams_dir, id+".csv"))

#     with lock:
#         if not method_id in STREAMS.keys(): STREAMS[method_id] = START_STREAM(streams_dir, dataset_dir)
#         METHODS[method_id]['streamer'] = str(STREAMS[method_id])

#     return make_response("Successfully extracted radiomics for case: " + scan_node["object.AccessionNumber"], 200)
