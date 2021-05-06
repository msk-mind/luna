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
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

import requests 
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("collect_result_segment.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    collect_result_segment_with_container(cohort_id, datastore_id, method_data)

def collect_result_segment_with_container(cohort_id, container_id, method_data, semaphore=0):
    """
    Using the container API interface, collect csv type output into a single container
    """
    output_datastore_id  = method_data.get("output_container")

    output_datastore = DataStore( cfg ).setNamespace(cohort_id)\
        .createDatastore(output_datastore_id, "parquet")\
        .setDatastore(output_datastore_id)
    input_container = DataStore( cfg ).setNamespace(cohort_id)\
        .setDatastore(container_id)

    try:
        df_list = []
        for tag in  method_data['input_tags']:
            node  = input_container.get("Radiomics", tag) 
            if node is None: continue
            df_tmp = pd.read_csv(node.data).astype('double', errors='ignore')
            df_tmp['meta_cohort_id']    = cohort_id
            df_tmp['meta_container_id'] = input_container._qualifiedpath
            df_tmp['meta_tag']          = tag
            df_tmp = df_tmp.set_index(["meta_container_id", "meta_tag"])
            df_list.append (df_tmp.loc[:, ~df_tmp.columns.str.contains('Unnamed')])
            
        df = pd.concat(df_list)

        logger.info (df)
        
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  output_datastore._namespace_id, output_datastore._name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"ResultSegment-{input_container._container_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_file)

        logger.info("Saved to : " + str(output_file))

        properties = {
            "rows": len(df),
            "columns": len(df.columns),
            "data": output_file
        }

    except Exception as e:
        input_container.logger.exception (f"{e}, stopping job execution...")
    else:
        output_node = Node("ResultSegment", f"slice-{input_container._container_id}", properties)
        output_datastore.put(output_node)
        
    finally:
        return semaphore + 1   



if __name__ == "__main__":
    cli()
