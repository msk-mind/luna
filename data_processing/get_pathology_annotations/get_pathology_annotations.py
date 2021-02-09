'''
This stands up the endpoints for getSlideIDs, getSlideIDS_case, and getPathologyAnnotation

How to run:
- start at the top level data_processing directory
- run: python3 -m data_processing.get_pathology_annotations.get_pathology_annotations -c {path to app config}

Example:
	python3 -m data_processing.get_pathology_annotations.get_pathology_annotations -c data_processing/get_pathology_annotations/app_config.yaml
'''

from flask import Flask, request, jsonify

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, MapType
from pyspark.sql.functions import  to_json

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time, click
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool
import re


app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
APP_CFG = "getPathologyAnnotations"


## regex for HobI and slide ids
slide_id_regex = re.compile("\d{6,}")
slide_hid_regex = re.compile("HobI\d{2}-\d{12,}")

PROJECT_MAPPING = const.PROJECT_MAPPING
ANNOTATION_TABLE_MAPPINGS = const.ANNOTATION_TABLE_MAPPINGS

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

# accepts both HobI and ImageIDs
@app.route('/mind/api/v1/getPathologyAnnotation/<string:project>/<string:id>/<string:annotation_type>/<labelset>', methods=['GET'])
def getPathologyAnnotation(annotation_type, project,id, labelset):
	
	pathology_root_path = app.config.get('pathology_root_path')

	spark = app.config.get('spark')

	if slide_hid_regex.match(id):
		slide_id = get_slide_id(id, "slide_hid")
	elif slide_id_regex.match(id):
		slide_id = id
	else:
		return "Invalid ID"

	if annotation_type not in ANNOTATION_TABLE_MAPPINGS:
		return "Illegal Annotation Type. This API supports \"regional\" or \"point\" annotations only"

	DATA_TYPE = ANNOTATION_TABLE_MAPPINGS[annotation_type]["DATA_TYPE"]
	GEOJSON_COLUMN = ANNOTATION_TABLE_MAPPINGS[annotation_type]["GEOJSON_COLUMN_NAME"]
	ANNOTATIONS_FOLDER = os.path.join(pathology_root_path, PROJECT_MAPPING[project])
	GEOJSON_TABLE_PATH = os.path.join(ANNOTATIONS_FOLDER , "tables", DATA_TYPE)

	row = spark.read.format("delta").load(GEOJSON_TABLE_PATH).where(f"slide_id='{slide_id}' and labelset='{labelset.upper()}' and latest=True")
	if row.count() == 0:
		return "No annotations match the provided query."
	geojson = row.select(to_json(GEOJSON_COLUMN).alias("val")).head()['val']
	return geojson


@click.command()
@click.option('-c',
              '--config_file',
              default="data_processing/get_pathology_annotations/app_config.yaml",
              type=click.Path(exists=True),
              help="path to config file for annotation API"
                   "See data_processing/get_pathology_annotations/app_config.yaml.template")
def cli(config_file):
	cfg = ConfigSet(name=APP_CFG, config_file=config_file)
	spark = SparkConfig().spark_session(config_name=APP_CFG, app_name="data_processing.mind.api")	
	pathology_root_path = cfg.get_value(path=APP_CFG+'::$.pathology[:1]["root_path"]')


	app.config['cfg'] = cfg
	app.config['spark'] = spark
	app.config['pathology_root_path'] = pathology_root_path




	app.run(host='0.0.0.0',port=5002, debug=True)



	

if __name__ == '__main__':
	cli()
	
