'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. for all dicoms, rescale into HU and optionally window
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, sys
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.utils           import get_method_data
from data_processing.common.Container       import Container
from data_processing.common.Node            import Node
from data_processing.common.config import ConfigSet

# From radiology.common
from data_processing.radiology.common.preprocess import window_dicoms

logger = init_logger("windowDicoms.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    method_data = get_method_data(cohort_id, method_id)
    window_dicom_with_container(cohort_id, container_id, method_data)

def window_dicom_with_container(cohort_id, container_id, method_data):
    """
    Using the container API interface, perform dicom CT preprocessing (windowing)
    """

    # Do some setup
    container   = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_id   = method_data.get("job_tag", "none")

    dicom_node  = container.get("dicom", method_data['dicom_input_tag']) # Only get origional dicoms from

    try:
        if dicom_node is None:
            raise ValueError("Dicom node not found")

        # Currently, store things at MIND_GPFS_DIR/data/<namespace>/<container name>/<method>/<schema>
        # Such that for every namespace, container, and method, there is one allocated path location
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, container._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = window_dicoms(
            dicom_paths = list(dicom_node.path.glob("*dcm")),
            output_dir = output_dir,
            params = method_data
        )
    
    except Exception:
        container.logger.exception ("Exception raised, stopping job execution.")
    else:

        output_node = Node('dicom', method_id, properties)
        container.add(output_node)
        container.saveAll()
    
if __name__ == "__main__":
    cli()
