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

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("collectCSV.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_id',    required=True)
def cli(cohort_id, container_id, method_id):
    method_data = get_method_data(cohort_id, method_id)
    collect_result_segment_with_container(cohort_id, container_id, method_data)

def collect_result_segment_with_container(cohort_id, container_id, method_data):
    """
    Using the container API interface, collect csv type output into a single container
    """

    # Do some setup
    container           = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    output_container    = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(method_data['output_container'])

    try:
        df_list = []
        for tag in  method_data['input_tags']:
            node  = container.get("Radiomics", tag) 
            if node is None: continue
            df_tmp = pd.read_csv(node.static_file).astype('double', errors='ignore')
            df_tmp['meta_cohort_id']    = cohort_id
            df_tmp['meta_container_id'] = container._qualifiedpath
            df_tmp['meta_tag']          = tag
            df_tmp = df_tmp.set_index(["meta_container_id", "meta_tag"])
            df_list.append (df_tmp.loc[:, ~df_tmp.columns.str.contains('Unnamed')])
            
        df = pd.concat(df_list)

        logger.info (df)
        
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", output_container._namespace_id, output_container._name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{container._container_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_file)

        logger.info("Saved to : " + str(output_file))

        properties = {
            "rows": len(df),
            "columns": len(df.columns),
            "file": output_file
        }

    except Exception:
        container.logger.exception ("Exception raised, stopping job execution.")
    else:
        output_node = Node("Parquet", f"slice-{container._container_id}", properties)
        output_container.add(output_node)
        output_container.saveAll()



if __name__ == "__main__":
    cli()
