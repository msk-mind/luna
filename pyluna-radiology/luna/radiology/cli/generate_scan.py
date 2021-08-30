'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. generate a volumentric image using ITK
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, logging
import click

# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore
from luna.common.Node            import Node
from luna.common.config import ConfigSet

# From radiology.common
from luna.radiology.common.preprocess import generate_scan

cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    generate_scan_with_container(cohort_id, datastore_id, method_data)

def generate_scan_with_container(cohort_id, container_id, method_data, semaphore=0):
    """
    Using the container API interface, generate a volumetric image for a given scan container
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    try:
         # Do some setup
        datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
        method_id   = method_data.get("job_tag", "none")
        
        dicom_node  = datastore.get("DicomSeries", method_data['dicom_input_tag']) # Only get origional dicoms from
        
        if dicom_node is None: raise ValueError("Dicom node not found")
        
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  datastore._namespace_id, datastore._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = generate_scan(
            dicom_path = dicom_node.data,
            output_dir = output_dir,
            params = method_data
        )
        
    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e
    else:
        output_node = Node("VolumetricImage", method_id, properties)
        datastore.put(output_node)
        
    finally:
        return semaphore + 1   

if __name__ == "__main__":
    cli()
