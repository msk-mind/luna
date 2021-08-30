'''
This stands up the endpoints for getSlideIDs, getSlideIDS_case, and getPathologyAnnotation

How to run:
- start at the top level luna directory
- run: python3 -m luna.get_pathology_annotations.get_pathology_annotations -c {path to app config}

Example:
        python3 -m luna.get_pathology_annotations.get_pathology_annotations -c luna/get_pathology_annotations/app_config.yaml
'''

from flask import Flask, request, jsonify

from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
from luna.common.Neo4jConnection import Neo4jConnection
from luna.common.config import ConfigSet
import luna.common.constants as const
from luna.common.DataStore import DataStore_v2

from pyarrow.parquet import read_table

import os, shutil, sys, importlib, json, yaml, subprocess, time, click
import re


app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
APP_CFG = "getPathologyAnnotations"


## regex for HobI and slide ids
slide_id_regex = re.compile("\d{6,}")
slide_hid_regex = re.compile("HobI\d{2}-\d{12,}")

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

        if slide_hid_regex.match(id):
                slide_id = get_slide_id(id, "slide_hid")
        elif slide_id_regex.match(id):
                slide_id = id
        else:
                return "Invalid ID"

        # point annots still uses spark-ETL-based organization
        ANNOTATIONS_FOLDER = os.path.join(pathology_root_path, project)

        if annotation_type == "point":
                DATA_TYPE = ANNOTATION_TABLE_MAPPINGS[annotation_type]["DATA_TYPE"]
                GEOJSON_TABLE_PATH = os.path.join(ANNOTATIONS_FOLDER , "tables", DATA_TYPE)

                row = read_table(GEOJSON_TABLE_PATH, columns = ["geojson"],
                                filters = [('slide_id', '=', f'{slide_id}')])

                if len(row) == 0:
                        return "No annotations match the provided query."
                geojson = json.dumps(row["geojson"].to_pylist()[0])
                return geojson
        elif annotation_type == "regional":
                store = DataStore_v2(os.path.join(ANNOTATIONS_FOLDER, "slides"))
                geojson_path = store.get (store_id=slide_id, namespace_id='CONCAT', data_type='RegionalAnnotationJSON', data_tag=labelset.upper())
                print(geojson_path)
                with open(geojson_path) as geojson_file:
                        return json.load(geojson_file)

        else:
                return "Illegal Annotation Type. This API supports \"regional\" or \"point\" annotations only"

@click.command()
@click.option('-c',
              '--config_file',
              default="luna/get_pathology_annotations/app_config.yaml",
              type=click.Path(exists=True),
              help="path to config file for annotation API"
                   "See luna/get_pathology_annotations/app_config.yaml.template")
def cli(config_file):
        cfg = ConfigSet(name=APP_CFG, config_file=config_file)
        pathology_root_path = cfg.get_value(path=APP_CFG+'::$.pathology[:1]["root_path"]')

        if os.environ.get("PATHOLOGY_ROOT_PATH", False): 
            logger.info("Overriding pathology_root_path!!!")
            pathology_root_path = os.environ.get("PATHOLOGY_ROOT_PATH")
            logger.info("pathology_root_path=" + pathology_root_path)

        app.config['cfg'] = cfg
        app.config['pathology_root_path'] = pathology_root_path


        app.run(host='0.0.0.0',port=5002, debug=False)



        

if __name__ == '__main__':
        cli()
        
