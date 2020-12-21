from flask import Flask, request, jsonify

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, MapType

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time, click
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
config_file = "data_processing/getPathologyAnnotations/config.yaml"
APP_CFG = "getPathologyAnnotations"


cfg = ConfigSet(name=APP_CFG, config_file=config_file)
spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="data_processing.mind.api")	
pathology_root_path = cfg.get_value(name=APP_CFG, jsonpath='$.pathology[:1]["root_path"]')


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
@app.route('/mind/api/v1/getPathologyAnnotation/<string:project>/<string:slide_hid>/<string:annotation_type>/<labelset>', methods=['GET'])
def getPathologyAnnotation(annotation_type, project,slide_hid,labelset):
	slide_id = get_slide_id(slide_hid, "slide_hid")

	ANNOTATIONS_FOLDER = os.path.join(pathology_root_path, project, "annotations")

	if annotation_type == "regional":
		GEOJSON_TABLE_PATH = ANNOTATIONS_FOLDER + "/table/regional_geojson"
	elif annotation_type == "point":
		GEOJSON_TABLE_PATH = ANNOTATIONS_FOLDER + "/table/point_refined_geojson"
	else:
		return None

	filepath = spark.read.format("delta").load(GEOJSON_TABLE_PATH).where(f"slide_id='{slide_id}' and labelset='{labelset}' and latest=True").first()["geojson_filepath"]

	print (filepath)

	with open(filepath) as f:
		geojson = f.read()
	return geojson


if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5002, debug=True)
