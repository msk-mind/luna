'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. for all dicoms, rescale into HU and optionally window
3. store results on HDFS and add metadata to the graph

'''
# General imports
import os, sys, subprocess, uuid
import click
from concurrent.futures import ProcessPoolExecutor

# Server imports
from flask import Flask, request, jsonify, render_template, make_response
from flask_restx import Api, Resource, fields

# From common
from data_processing.common.config          import ConfigSet
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data
from data_processing.pathology.common.utils import pretile_scoring

logger = init_logger("tile_slide_service.log")

def tile_slide_with_container(cohort_id, container_id, method_id, job_id=None):
    # Eventually these will come from a cfg file, or somewhere else
    container_params = {
        'GRAPH_URI':  os.environ['GRAPH_URI'],
        'GRAPH_USER': "neo4j",
        'GRAPH_PASSWORD': "password",
        'MINIO_URI': 'pllimsksparky1:8002',
        'MINIO_USER': "mind",
        'MINIO_PASSWORD': 'm2eN6TVfZOaeqiuieOGf'
    }

    # Do some setup
    container   = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_data = get_method_data(cohort_id, method_id) 
    input_node  = container.get("wsi") # Only get origional dicoms from
    
    # Currently, store things at MIND_GPFS_DIR/data/<namespace>/<container name>/<method>/<schema>
    # Such that for every namespace, container, and method, there is one allocated path location
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, container_id, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    logger.info("Allocated output directory at %s", output_dir)
    tile_addresses = pretile_scoring(container_id, str(input_node.path),  256, 20, 16, 16, "", {'otsu': -1, 'purple': -1, 'annotation': False}, "", "")
    print (len(tile_addresses))
    # container.add(output_node)
    # container.saveAll()

    if job_id: logger.info(f"Job {job_id} finished successfully.")
    
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")
VERSION      = "branch:"+subprocess.check_output(["git","rev-parse" ,"--abbrev-ref", "HEAD"]).decode('ascii').strip()
HOSTNAME     = os.environ["HOSTNAME"]
PORT         = int(cfg.get_value("APP_CFG::tile_wsi_port"))
NUM_PROCS    = int(cfg.get_value("APP_CFG::tile_wsi_processes"))

app = Flask(__name__)
api = Api(app, version=VERSION, title='Window Dicom Service', description='Worker for window_dicom()', ordered=True)
executor = ProcessPoolExecutor(NUM_PROCS) 

@api.route('/mind/api/v1/window_dicom/<cohort_id>/<container_id>/<method_id>/submit', methods=['GET'])
class API_window_dicom(Resource):
    def get(self, cohort_id, container_id, method_id):
        """Submit job"""
        job_id = str(uuid.uuid4())
        future = executor.submit (container_window_dicom, cohort_id, container_id, method_id, job_id)
        return f"Submitted job {job_id} with future {future}"

@click.command()
@click.argument('run')
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(run, cohort_id, container_id, method_id):
    tile_slide_with_container(cohort_id, container_id, method_id)

if __name__ == '__main__':
    # Setup App/Api
    if sys.argv[1] == "start":
        logger.info(f"Running in API mode")
        logger.info(f"Starting worker on {HOSTNAME}:{PORT}")
        app.run(host=HOSTNAME,port=PORT, debug=False)
    if sys.argv[1] == "run":
        logger.info(f"Running as in CLI mode")
        cli()
