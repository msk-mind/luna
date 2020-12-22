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
spark  = SparkConfig().spark_session(const.APP_CFG, "data_processing.radiology.api.5003")
conn   = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
lock   = threading.Lock()

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

STREAMS = {}
METHODS = {}
HOST = os.environ['HOSTNAME']

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

# Query all the scan ids in a cohort
# Return by neo4j id type
@app.route('/mind/api/v1/getScanIDs/<cohort_id>', methods=['GET'])
def getScanIDs(cohort_id):
    res = conn.query(f"""MATCH (px:patient)-[:PROXY]-(proxy)-[:HAS_CASE]-(case:case)-[:HAS_SCAN]-(scan:scan) where px.cohort='{cohort_id}' and px.active=True and case.cohort='{cohort_id}' and case.active=True RETURN id(scan) as id""")
    return jsonify([rec.data()['id'] for rec in res])

# Generate a scan given node id
# Adds a mhd node to the scan ID upon successful completition
@app.route('/mind/api/v1/generateScan/<id>', methods=['GET'])
def generateScan(id):

    scan_node = conn.query(f""" MATCH (object:scan) WHERE id(object)={id} RETURN object""")
    if not scan_node: return make_response("Send a node ID that is not a scan ID", 500)

    scan_node   = scan_node[0].data()['object']
    path        = scan_node['path']
    file_ext    = 'mhd'

    input_dir, filename  = os.path.split(path)
    input_dir = input_dir[input_dir.index("/"):]

    output_dir = os.path.join("/gpfs/mskmindhdp_emc/data/", scan_node['cohort'], "scans", scan_node['SeriesInstanceUID'])
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
    record_uuid = "-".join(["SCAN", dirhash(input_dir, "sha256")])

    result = conn.query(f""" MATCH (object:scan) WHERE id(object)={id} MERGE (result:mhd{{ record_uuid:"{record_uuid}", path:"{outFileName}", z_dim: {n_slices} }}) MERGE (object)-[:HAS_DATA]-(result) RETURN result""")

    return make_response("Successfully generated volumetric image", 200)

# Extract radiomics
@app.route('/mind/api/v1/radiomics/<method_id>/<id>', methods=['GET'])
def extractRadiomics(method_id, id):
    if not method_id in METHODS.keys():
        return make_response("No extractor configured for " + method_id)

    scan_node = conn.query(f"""
        MATCH (object:scan)-[:HAS_DATA]-(image:mhd)
        MATCH (object:scan)-[:HAS_DATA]-(label:mha)
        WHERE id(object)={id}
        RETURN object.cohort, object.AccessionNumber, image.path, label.path"""
    )

    if not scan_node:
        return make_response("Scan is not ready for radiomics (missing annotation?)", 500)

    scan_node = scan_node[0].data()

    JOB_CONFIG = METHODS[method_id]

    config      = JOB_CONFIG['config']
    streams_dir = JOB_CONFIG['streams_dir']
    dataset_dir = JOB_CONFIG['dataset_dir']

    extractor = featureextractor.RadiomicsFeatureExtractor(**config['params'])

    try:
        result = extractor.execute(scan_node["image.path"].split(':')[-1], scan_node["label.path"].split(':')[-1])
    except Exception as e:
        return make_response(str(e), 200)

    sers = pd.Series(result)
    sers["AccessionNumber"] = scan_node["object.AccessionNumber"]
    sers["config"]          = config
    sers.to_frame().transpose().to_csv(os.path.join(streams_dir, id+".csv"))

    with lock:
        if not method_id in STREAMS.keys(): STREAMS[method_id] = START_STREAM(streams_dir, dataset_dir)
        METHODS[method_id]['streamer'] = str(STREAMS[method_id])

    return make_response("Successfully extracted radiomics for case: " + scan_node["object.AccessionNumber"], 200)


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5003, threaded=True, debug=False)
