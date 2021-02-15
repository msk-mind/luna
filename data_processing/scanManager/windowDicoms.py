'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. for all dicoms, rescale into HU and optionally window
3. store results on HDFS and add metadata to the graph

'''

# General imports
import sys, os, glob
import click
from checksumdir import dirhash

# From common
from data_processing.common.Node       import Node
from data_processing.common.Container  import Container
from data_processing.common.utils      import get_method_data 

logger = init_logger("windowDicoms.log")

# Special libraries
from pydicom import dcmread
import numpy as np



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
    
    # Currently, store things at MIND_GPFS_DIR/data/<namespace>/<container name>/<method>/<schema>
    # Such that for every namespace, container, and method, there is one allocated path location
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, container._name, method_id)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Scale and clip each dicom, and save in new directory
    container.logger.info("Processing %s dicoms!", len(list(input_node.path.parent.glob("*dcm"))))
    for dcm in input_node.path.parent.glob("*dcm"):
        ds = dcmread(dcm)
        hu = ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept
        if method_data['window']:
            hu = np.clip( hu, method_data['window.low_level'], method_data['window.high_level']   )
        ds.PixelData = hu.astype(ds.pixel_array.dtype).tobytes()
        ds.save_as (os.path.join( output_dir, dcm.stem + ".cthu.dcm"  ))

    # Prepare metadata and commit
    record_type = "dicom"
    record_name = method_data['output_name']
    record_properties = {
        "RescaleSlope":ds.RescaleSlope, 
        "RescaleIntercept":ds.RescaleIntercept, 
        "units":"HU", 
        "path":output_dir, 
        "hash":dirhash(output_dir, "sha256")
    }

    output_node = Node(record_type, record_name, record_properties)

    container.add(output_node)
    container.saveAll()
    
if __name__ == "__main__":
    cli()
