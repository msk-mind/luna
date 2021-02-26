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
from data_processing.common.utils           import get_method_data
from data_processing.common.Container       import Container
from data_processing.common.Node            import Node
from data_processing.common.config import ConfigSet

from pymongo import MongoClient

# From radiology.common
from data_processing.radiology.common.preprocess   import extract_radiomics

logger = init_logger("extractRadiomics.log")
cfg = ConfigSet("CONTAINER_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    method_data = get_method_data(cohort_id, method_id)
    extract_radiomics_with_container(cohort_id, container_id, method_data)

def extract_radiomics_with_container(cohort_id, container_id, method_data):
    if method_data.get("job_id", False):
        job_db = MongoClient(cfg.get_value("APP_CFG::MONGODB_URI")).db.jobs
        job_db.insert_one ( {"job_id": method_data.get("job_id"), "status": "running" } )

    # Do some setup
    container   = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_id   = method_data.get("job_tag", "none")

    image_node  = container.get("mhd", method_data['image_input_tag']) 
    label_node  = container.get("mha", method_data['label_input_tag'])

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        if label_node is None:
            raise ValueError("Label node not found")
        
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = extract_radiomics(
            image_path = str(next(image_node.path.glob("*.mhd"))),
            label_path = str(label_node.path),
            output_dir = output_dir,
            params     = method_data
        )
    except Exception as e:
        container.logger.exception ("Exception raised, stopping job execution.")
        if method_data.get("job_id", False):
            job_db.insert_one ( {"job_id": method_data.get("job_id"), "status": "failed" } )
    else:
        output_node = Node("radiomics", method_id, properties)
        container.add(output_node)
        container.saveAll()
        if method_data.get("job_id", False):
            job_db.insert_one ( {"job_id": method_data.get("job_id"), "status": "succeeded" })

    finally:
        if method_data.get("job_id", False):
            job_db.insert_one ( {"job_id": method_data.get("job_id"), "status": "done" })



if __name__ == "__main__":
    cli()
