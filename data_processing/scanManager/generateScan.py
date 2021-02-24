'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. generate a volumentric image using ITK
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, sys
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.utils           import get_method_data
from data_processing.common.Container       import Container
from data_processing.common.Node            import Node
from data_processing.common.config import ConfigSet

# From radiology.common
from data_processing.radiology.common.preprocess import generate_scan

logger = init_logger("generateScan.log")
cfg = ConfigSet("CONTAINER_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    method_data = get_method_data(cohort_id, method_id)
    generate_scan_with_container(cohort_id, container_id, method_data)

def generate_scan_with_container(cohort_id, container_id, method_data):

    # Do some setup
    container   = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_id   = method_data.get("job_tag", "none")
    
    input_node  = container.get("dicom", method_data['input_name']) # Only get origional dicoms from

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    properties = generate_scan(
        dicom_path = str(input_node.path),
        output_dir = output_dir,
        params = method_data
    )
    
    if properties is None: return

    output_node = Node(method_data['file_ext'], method_id, properties)

    container.add(output_node)
    container.saveAll()




if __name__ == "__main__":
    cli()
