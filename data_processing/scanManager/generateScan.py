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
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data
from data_processing.radiology.common.utils    import generate_scan

logger = init_logger("generateScan.log")

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
    input_node  = container.get("dicom", method_data['input_name']) # Only get origional dicoms from

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_node = generate_scan(
        name = method_id,
        dicom_path = str(input_node.path),
        output_dir = output_dir,
        params = method_data
    )
    
    if output_node is None: return

    container.add(output_node)
    container.saveAll()




if __name__ == "__main__":
    cli()
