from flask import Flask, request, jsonify

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, MapType

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
# spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")

# ==================================================================================================
# Service functions
# ==================================================================================================
def get_slide_id( alias_id, alias_name ):
	conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")
	results = conn.query(f'''MATCH (n:slide) where n.{alias_name}="{alias_id}" RETURN n''')
	if len(results)==0: return "NULL"
	return results[0].data()['n']['slide_id']

"""
curl http://<server>:5001/mind/api/v1/datasets/MY_DATASET
"""
@app.route('/mind/api/v1/getPathologyAnnotation/<string:project>/<string:slide_hid>/<labelset>', methods=['GET'])
def getPathologyAnnotation(project,slide_hid,labelset):
	slide_id = get_slide_id(slide_hid, "slide_hid")

	spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.getPathologyAnnotation")

	ANNOTATIONS_FOLDER = os.path.join("/gpfs/mskmindhdp_emc/data/pathology", project, "annotations")
	GEOJSON_TABLE_PATH = ANNOTATIONS_FOLDER + "/table/geojson"

	filepath = spark.read.format("delta").load(GEOJSON_TABLE_PATH).where(f"slide_id='{slide_id}' and labelset='{labelset}' and latest=True").first()["geojson_filepath"]

	print (filepath)

	with open(filepath) as f:
		geojson = f.read()
	return geojson

if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'],port=5002, debug=True)
