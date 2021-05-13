'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to a volumentric image and annotation (label) files
2. resample image and segmentation, and save voxels as a 3d numpy array (.npy)
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, logging
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

# From radiology.common
from data_processing.radiology.common.preprocess   import extract_voxels

cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    extract_voxels_with_container(cohort_id, datastore_id, method_data)

def extract_voxels_with_container(cohort_id: str, container_id: str, method_data: dict, semaphore=0):
    """
    Using the container API interface, extract voxels for a given scan container
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    # Do some setup
    datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
    method_id   = method_data.get("job_tag", "none")

    image_node  = datastore.get("VolumetricImage", method_data['image_input_tag']) 
    label_node  = datastore.get("VolumetricLabel", method_data['label_input_tag'])

    try:
        if image_node is None: raise ValueError("Image node not found")
        if label_node is None: raise ValueError("Label node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  datastore._namespace_id, datastore._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = extract_voxels(
            image_path = image_node.data,
            label_path = label_node.data,
            output_dir = output_dir,
            params     = method_data
        )

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e
    else:
        output_node = Node("Voxels", method_id, properties)
        datastore.put(output_node)
        
    finally:
        return semaphore + 1   

if __name__ == "__main__":
    cli()
