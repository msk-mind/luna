import os, json, sys
import itk
import click
from filehash import FileHash
import pandas as pd

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.GraphEnum import Node
from data_processing.common.custom_logger import init_logger

from radiomics import featureextractor  # This module is used for interaction with pyradiomics

logger = init_logger("extractRadiomics.log")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    logger.info("Invocation: " + str(sys.argv))

    properties = {}
    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    properties['Namespace'] = cohort_id
    properties['MethodID']  = method_id

    with open(f'{method_id}.json') as json_file:
        method_config = json.load(json_file)['params']

    input_nodes = conn.query(f"""
        MATCH (object:scan)-[:HAS_DATA]-(image:mhd)
        MATCH (object:scan)-[:HAS_DATA]-(label:mha)
        WHERE id(object)={container_id}
        RETURN object.SeriesInstanceUID, image.path, label.path"""
    )

    if not input_nodes or len (input_nodes)==0:
        logger.error ("Scan is not ready for radiomics (missing annotation?)")
        logger.info (f"""MATCH (object:scan)-[:HAS_DATA]-(image:mhd) MATCH (object:scan)-[:HAS_DATA]-(label:mha) WHERE id(object)={container_id} RETURN object.SeriesInstanceUID, image.path, label.path""")
        return 

    input_data = input_nodes[0].data()

    print (input_data)

    extractor = featureextractor.RadiomicsFeatureExtractor(**method_config)

    try:
        result = extractor.execute(input_data["image.path"].split(':')[-1], input_data["label.path"].split(':')[-1])
    except Exception as e:
        logger.error (str(e))
        return

    output_dir = os.path.join("/gpfs/mskmindhdp_emc/data/COHORTS", cohort_id, "scans", input_data['object.SeriesInstanceUID'], method_id+".csv")

    sers = pd.Series(result)

    sers.to_frame().transpose().to_csv(output_dir)

    logger.info("Saving to " + output_dir)

    properties['RecordID'] = "RAD" + "-" + str(FileHash('sha256').hash_file(output_dir))
    properties['path'] = output_dir

    n_meta = Node("radiomics", properties=properties)

    conn.query(f""" 
        MATCH (sc:scan) WHERE id(sc)={container_id}
        MERGE (da:{n_meta.create()})
        MERGE (sc)-[:HAS_DATA]->(da)"""
    )

    logger.info ("Successfully extracted radiomics for scan: " + input_data["object.SeriesInstanceUID"])



if __name__ == "__main__":
    cli()
