'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to a volumentric image and annotation (label) files
2. extract radiomics features into a vector (csv)
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, sys
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

# From radiology.common
from data_processing.radiology.common.preprocess   import extract_radiomics

logger = init_logger("extract_radiomics.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    extract_radiomics_with_container(cohort_id, datastore_id, method_data)

def extract_radiomics_with_container(cohort_id, container_id, method_data, semaphore=0):
    """
    Using the container API interface, extract radiomics for a given scan container
    """

    # Do some setup
    datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
    method_id   = method_data.get("job_tag", "none")

    try:
        image_node  = datastore.get("VolumetricImage",    method_data['image_input_tag'])

        if method_data.get("usingPertubations", False):
            label_node  = datastore.get("VolumetricLabelSet", method_data['label_input_tag'])
        else: 
            label_node  = datastore.get("VolumetricLabel", method_data['label_input_tag'])

        if image_node is None: raise ValueError("Image node not found")
        if label_node is None: raise ValueError("Label node not found")
        
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  datastore._namespace_id, datastore._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = extract_radiomics(
            image_path = image_node.data,
            label_path = label_node.data,
            output_dir = output_dir,
            params     = method_data
        )
    except Exception as e:
        datastore.logger.exception (f"{e}, stopping job execution...")
    else:
        if properties:
            output_node = Node("Radiomics", method_id, properties)
            datastore.put(output_node)
            
    finally:
        return semaphore + 1   

if __name__ == "__main__":
    cli()
