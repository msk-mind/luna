'''
Created: January 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to radiomics results
2. prepare a parquet table to save for this container
3. export table to publically available path on ess

Job parameters:
"params": {
    "output_dir": <string> # Destination directory
    "input":      <string> # Used to match a specific radiomics result
}
'''

# General imports
import os, json, sys
import click
import pandas as pd

# From common
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger   import init_logger
from data_processing.common.utils import get_method_data
import data_processing.common.constants as const

# Specaialized libraries to make parquet table
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("saveRadiomics.log")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    logger.info("Invocation: " + str(sys.argv))

    conn = Neo4jConnection(uri=os.environ["GRAPH_URI"], user="neo4j", pwd="password")

    method_data = get_method_data(cohort_id, method_id)

    input_method_id = method_data['input_name']

     # Get relevant data, matching MethodID
    input_nodes = conn.query(f"""
        MATCH (px:patient)-[:HAS_CASE]-(case)-[:HAS_SCAN]-(scan:scan)-[:HAS_DATA]-(results:radiomics)
        WHERE id(scan)={container_id} AND results.name='{input_method_id}'
        RETURN px.name, case.AccessionNumber, scan.SeriesInstanceUID, results.path, results.hash"""
    )

    if not input_nodes or len (input_nodes)==0:
        logger.error ("Query failed!!")
        return 

    input_data = input_nodes[0].data()

    logger.info (input_data)

    output_dir  = os.path.join(const.PUBLIC_DIR, method_data['output_dir'])
    output_file = os.path.join(output_dir, input_data['results.hash'] + ".flatten.parquet")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # Get Results package
    df = pd.read_csv(input_data['results.path'] + "/radiomics-out.csv").astype('double', errors='ignore')

    # Add ID hierachy information
    df['meta_patient_id']         = input_data['px.name']
    df['meta_accession_number']   = input_data['case.AccessionNumber']

    # Add operational information
    df['meta_cohort_id'] = cohort_id
    df['meta_container_id'] = container_id
    df['meta_method_id'] = method_id
    df['meta_input'] = input_method_id

    # Cleanup unnamed columns
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    # Create table part
    table = pa.Table.from_pandas(df)

    # Write!
    pq.write_table(table, output_file)

if __name__ == "__main__":
    cli()
