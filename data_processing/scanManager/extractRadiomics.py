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
import pandas as pd
from filehash import FileHash

# From common
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Node       import Node

# Specialized library to extract radiomics
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

logger = init_logger("extractRadiomics.log")

def get_container_data(container_id):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    input_nodes = conn.query(f"""
        MATCH (object:scan)-[:HAS_DATA]-(image:mhd)
        MATCH (object:scan)-[:HAS_DATA]-(label:mha)
        WHERE id(object)={container_id}
        RETURN object.SeriesInstanceUID, image.path, label.path"""
    )

    if not input_nodes or len (input_nodes)==0:
        logger.error ("Scan is not ready for radiomics (missing annotation?)")
        return [] 
    else:
        return input_nodes[0].data()

def get_method_data(method_id):
    with open(f'{method_id}.json') as json_file:
        method_config = json.load(json_file)['params']
    return method_config

def add_container_data(container_id, n_meta):
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    res = conn.query(f""" 
        MATCH (sc:scan) WHERE id(sc)={container_id}
        MERGE (da:{n_meta.get_create_str()})
        MERGE (sc)-[:HAS_DATA]->(da)"""
    )

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    print("Invocation: " + str(sys.argv))

    properties = {}
    properties['Namespace'] = cohort_id
    properties['MethodID']  = method_id

    input_data = get_container_data(container_id) 
    method_data = get_method_data(method_id) 

    print (input_data)
    print (method_data)

    extractor = featureextractor.RadiomicsFeatureExtractor(**method_data)

    try:
        result = extractor.execute(input_data["image.path"].split(':')[-1], input_data["label.path"].split(':')[-1])
    except Exception as e:
        logger.error (str(e))
        return

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data/COHORTS", cohort_id, "scans", input_data['object.SeriesInstanceUID'])

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, method_id+".csv")

    sers = pd.Series(result)

    print("Saving to " + output_filename)

    sers.to_frame().transpose().to_csv(output_filename)

    record_name = "RAD" + "-" + str(FileHash('sha256').hash_file(output_filename))

    properties['path'] = output_filename 

    n_meta = Node("radiomics", record_name, properties=properties)

    add_container_data(container_id, n_meta)

    print ("Successfully extracted radiomics for scan: " + input_data["object.SeriesInstanceUID"])

if __name__ == "__main__":
    cli()
