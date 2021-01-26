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
        MATCH (container)-[:HAS_DATA]-(image:mhd)
        MATCH (container)-[:HAS_DATA]-(label:mha)
        WHERE id(container)={container_id}
        RETURN container.QualifiedPath, container.name, image.path, label.path"""
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
        MATCH (container) WHERE id(container)={container_id}
        MERGE (da:{n_meta.get_create_str()})
        MERGE (container)-[:HAS_DATA]->(da)"""
    )

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    # =========================================================================================
    # Setup/Initalize
    logger.info("Invocation: " + str(sys.argv))

    properties = {}
    properties['Namespace'] = cohort_id
    properties['MethodID']  = method_id

    # New empty container - as anything that has data, manage things at the container level? 
    # What we're managing, need to be more clear
    # Container abstraction on a good track!
    # Job vs. task, lifecycle
    # Jobs are run on containers, but
    # Containers as a way to locate data/metadata as it relates to the abstract
    # Future: 
->  container = Container().setNamespace(cohort).lookupAndAttach()

    # Get
    method_data = get_method_data(method_id) 

    logger.info (input_data)
    logger.info (method_data)

    # TODO: down the road
->  container.run("something")

    # =========================================================================================
    # Execute
->  images = container.get(type="mhd", namespace="")
        -> 1 query the node's metadata
        -> 2. (optionally) if non-local file, also download into a tmp directory
->  labels = container.get("mha")

    if images is None or labels is None:
        logger.error ("Cannot find a label and/or image")
        return


    extractor = featureextractor.RadiomicsFeatureExtractor(**method_data)

    try:
        result = extractor.execute(images["path"], labels["path"])
    except Exception as e:
        logger.error (str(e))
        return

    # Data just goes under namespace/name
    # TODO: This path is really not great, but works for now
    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data/COHORTS", cohort_id, input_data['container.name'])
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, method_id+".csv")
    logger.info("Saving to " + output_filename)

    sers = pd.Series(result)
    sers.to_frame().transpose().to_csv(output_filename)

    record_name = "RAD" + "-" + str(FileHash('sha256').hash_file(output_filename))
    properties['path'] = output_filename 

    # =========================================================================================
    # Finalize/save
    # Add a node to the container
->  container.add(Node("radiomics", record_name, properties=properties))
    # Commit results (run neccessary cypher queries)
->  container.save()
        -> 1 create query with node's metadata
        -> 2. (optionally) if non-local file, also push/upload from tmp to some remote storage if it exists

    logger.info ("Successfully extracted radiomics for container: " + input_data["container.QualifiedPath"])

if __name__ == "__main__":
    cli()
