'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to radiomics results
2. prepare a parquet table to save for this container
3. export table to publically available path on ess

'''

# General imports
import os, json, sys
import click
import pandas as pd

# From common
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger   import init_logger

# Specaialized libraries to make parquet table
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("flattenRadiomics.log")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    logger.info("Invocation: " + str(sys.argv))

    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    with open(f'{method_id}.json', 'r') as json_file:
        method_config = json.load(json_file)['params']
        
    results_to_flatten = method_config['method_name']

    input_nodes = conn.query(f"""
        MATCH (px:patient)-[:HAS_CASE]-(case)-[:HAS_SCAN]-(scan:scan)-[:HAS_DATA]-(results:radiomics)
        WHERE id(scan)={container_id} AND results.MethodID='{results_to_flatten}'
        RETURN px.PatientID, case.AccessionNumber, scan.SeriesInstanceUID, results.path"""
    )

    output_dir  = os.path.join("/gpfs/mskmind_ess/mind_public", method_config['output_dir'])
    output_file = os.path.join(output_dir, f"{container_id}.flatten.parquet")

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    if not input_nodes or len (input_nodes)==0:
        logger.error ("Query failed!!")
        return 

    input_data = input_nodes[0].data()

    logger.info (input_data)

    # Get Results package
    df = pd.read_csv(input_data['results.path'])

    # Add ID hierachy information
    df['PatientID']         = input_data['px.PatientID']
    df['AccessionNumber']   = input_data['case.AccessionNumber']
    df['SeriesInstanceUID'] = input_data['scan.SeriesInstanceUID']

    # Add operational information
    df['cohort_id'] = cohort_id
    df['container_id'] = container_id
    df['method_id'] = method_id

    # Cleanup unnamed columns
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    # Create table part
    table = pa.Table.from_pandas(df)

    # Write!
    pq.write_table(table, output_file)

if __name__ == "__main__":
    cli()
