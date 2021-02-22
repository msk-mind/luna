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
from data_processing.common.Node  import Node
from data_processing.common.utils      import get_method_data
from data_processing.radiology.common.preprocess import window_dicoms

logger = init_logger("windowDicoms.log")

def container_window_dicom(cohort_id, container_id, method_id, job_id=None):
    # Eventually these will come from a cfg file, or somewhere else
    container_params = {
        'GRAPH_URI':  os.environ['GRAPH_URI'],
        'GRAPH_USER': "neo4j",
        'GRAPH_PASSWORD': "password",
        'MINIO_URI': 'localhost:8002',
        'MINIO_USER': "mind",
        'MINIO_PASSWORD': 'm2eN6TVfZOaeqiuieOGf'
    }

    # Do some setup
    container   = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_data = get_method_data(cohort_id, method_id) 
    input_node  = container.get("dicom", 'init-scans') # Only get origional dicoms from
    
    # Currently, store things at MIND_GPFS_DIR/data/<namespace>/<container name>/<method>/<schema>
    # Such that for every namespace, container, and method, there is one allocated path location
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    logger.info("Allocated output directory at %s", output_dir)

    try:
        properties = window_dicoms(
            dicom_paths = list(input_node.path.glob("*dcm")),
            output_dir = output_dir,
            params = method_data
        )
    except Exception as e:
        if job_id: logger.warning(f"Job {job_id} finished with errors: {e}")
        return
    output_node = Node("dicom", method_id, properties)
 
    logger.info("Preparing")

    container.add(output_node)
    container.saveAll()

    if job_id: logger.info(f"Job {job_id} finished successfully.")
    
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")
VERSION      = "branch:"+subprocess.check_output(["git","rev-parse" ,"--abbrev-ref", "HEAD"]).decode('ascii').strip()
HOSTNAME     = os.environ["HOSTNAME"]
PORT         = int(cfg.get_value("APP_CFG::window_dicoms_port"))
NUM_PROCS    = int(cfg.get_value("APP_CFG::window_dicoms_processes"))

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


@api.route('/service/health', methods=['GET'])
class API_window_dicom(Resource):
    def get(self):
        return make_response("Alive.")

@click.command()
@click.argument('run')
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(run, cohort_id, container_id, method_id):
    container_window_dicom(cohort_id, container_id, method_id)

if __name__ == '__main__':
    # Setup App/Api
    if sys.argv[1] == "start":
        logger.info(f"Running in API mode")
        logger.info(f"Starting worker on {HOSTNAME}:{PORT}")
        app.run(host=HOSTNAME,port=PORT, debug=True)
    if sys.argv[1] == "run":
        logger.info(f"Running as in CLI mode")
        cli()
