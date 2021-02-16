'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to a volumentric image and annotation (label) files
2. extract radiomics features into a vector (csv)
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, sys
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data
from data_processing.radiology.common.utils   import extract_radiomics

logger = init_logger("extractRadiomics.log")

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
    label_node  = container.get("mha", method_data['label_input_name'])

    # Data just goes under namespace/name
    # TODO: This path is really not great, but works for now
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_node = extract_radiomics(
        name = method_id,
        image_path = str(next(image_node.path.glob("*.mhd"))),
        label_path = str(label_node.path),
        output_dir = output_dir,
        params     = method_data
    )

    if output_node is None: return

    container.add(output_node)
    container.saveAll()



if __name__ == "__main__":
    cli()
