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
from checksumdir import dirhash

# From common
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Node       import Node
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data 

logger = init_logger("extractRadiomics.log")

# Specialized library to extract radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
import pandas as pd

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):

    # Eventually these will come from a cfg file, or somewhere else
    container_params = {
        'GRAPH_URI':  os.environ['GRAPH_URI'],
        'GRAPH_USER': "neo4j",
        'GRAPH_PASSWORD': "password"
    }

    # Do some setup
    container   = Container( container_params ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_data = get_method_data(cohort_id, method_id) 

    image_node  = container.get("mhd", method_data['image_input_name']) 
    label_node  = container.get("mha", method_data['image_output_name'])

    extractor = featureextractor.RadiomicsFeatureExtractor(**method_data)

    try:
        result = extractor.execute(image_node.get_object("*.mhd"), label_node.get_object("*.mha"))
    except Exception as e:
        container.logger.error ("Extraction failed!!!")
        container.logger.error (str(e))
        return

    # Data just goes under namespace/name
    # TODO: This path is really not great, but works for now
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, method_id+".csv")

    container.logger.info("Saving to " + output_filename)
    sers = pd.Series(result)
    sers.to_frame().transpose().to_csv(output_filename)

    # Prepare metadata and commit
    record_type = "radiomics"
    record_name = method_data['output_name']
    record_properties = {
        "path":output_dir, 
        "hash":dirhash(output_dir, "sha256")
    }

    output_node = Node(record_type, record_name, record_properties)

    container.add(output_node)
    container.saveAll()

if __name__ == "__main__":
    cli()
