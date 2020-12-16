from flask import Flask, request, jsonify

from   data_processing.common.CodeTimer import CodeTimer
from   data_processing.common.custom_logger import init_logger
from   data_processing.common.config import ConfigSet
from   data_processing.common.sparksession import SparkConfig
from   data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const

import os, shutil, sys, importlib, json, yaml, subprocess, time

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
# spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")

cfg = ConfigSet(name="APP_CFG",  config_file='/app/config.yaml')

# ==================================================================================================
# Service functions
# ==================================================================================================
def get_slide_id( alias_id, alias_name ):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    results = conn.query(f'''MATCH (n:slide) where n.{alias_name}="{alias_id}" RETURN n''')
    if len(results)==0: return "NULL"
    return results[0].data()['n']['slide_id']

"""
Return slide (slides) given partial match to an slide ID alias
"""
@app.route('/mind/api/v1/getSlideIDs/<string:input_id>', methods=['GET'])
def getSlideIDs(input_id):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    results = conn.query(f'''MATCH (n:slide) WHERE (any (prop in keys(n) where n[prop] contains '{input_id}')) RETURN n''')
    return jsonify([res.data()['n'] for res in results])
"""
Return slide (slides) given hobS case ID
"""
@app.route('/mind/api/v1/getSlideIDs/case/<string:input_id>', methods=['GET'])
def getSlideIDs_case(input_id):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
    results = conn.query(f'''MATCH (m:accession)-[HAS_SLIDE]-(n:slide) WHERE m.case_hid='{input_id}' RETURN n''')
    return jsonify([res.data()['n'] for res in results])
"""
curl http://<server>:5001/mind/api/v1/datasets/MY_DATASET
"""
@app.route('/mind/api/v1/getPathologyAnnotation/<string:project>/<string:slide_hid>/<labelset>', methods=['GET'])
def getPathologyAnnotation(project,slide_hid,labelset):
    slide_id = get_slide_id(slide_hid, "slide_hid")

    spark = SparkConfig().spark_session(config_name="APP_CFG", app_name="data_processing.getPathologyAnnotation")

    ANNOTATIONS_FOLDER = os.path.join(os.environ["HDFS_URI"] + "/data/pathology", project, "annotations")

    GEOJSON_TABLE_PATH = ANNOTATIONS_FOLDER + "/table/geojson"

    filepath = spark.read.format("delta").load(GEOJSON_TABLE_PATH).where(f"slide_id='{slide_id}' and labelset='{labelset}' and latest=True").first()["geojson_filepath"]

    print (filepath)

    with open(filepath) as f:
        geojson = f.read()
    return geojson

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5002, debug=True)
