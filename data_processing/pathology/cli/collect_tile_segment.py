'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a slide (container) ID
1. resolve the path to the WsiImage and TileLabels
2. perform various scoring and labeling to tiles
3. save tiles as a parquet file with schema [address, coordinates, *scores, *labels ]

Example:
python3 -m data_processing.pathology.cli.collect_tiles \
    -c TCGA-BRCA \
    -s tcga-gm-a2db-01z-00-dx1.9ee36aa6-2594-44c7-b05c-91a0aec7e511 \
    -m data_processing/pathology/cli/example_collect_tiles.json
'''

# General imports
import os, json, sys
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.utils           import get_method_data
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

import requests
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("collect_tiles.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, container_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    collect_tile_results_with_container(cohort_id, container_id, method_data)

def collect_tile_results_with_container(cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
    """
    input_tile_data_id   = method_data.get("input_label_tag")
    output_container_id  = method_data.get("output_container")

    output_container = DataStore( cfg ).setNamespace(cohort_id).createContainer(output_container_id, "parquet").setContainer(output_container_id)
    input_container  = DataStore( cfg ).setNamespace(cohort_id).setContainer(container_id)

    image_node  = input_container.get("TileImages", input_tile_data_id) 

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        df = pd.read_csv(image_node.aux)
        df.loc[:,"data_path"]     = image_node.data
        df.loc[:,"object_bucket"] = image_node.properties['object_bucket']
        df.loc[:,"object_path"]   = image_node.properties['object_folder'] + "/tiles.slice.pil"
        df.loc[:,"id_slide_container"] = input_container._name

        df = df.set_index(["id_slide_container", "address"])
        logger.info(df)

        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", output_container._namespace_id, output_container._name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{input_container._container_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_file)

        logger.info("Saved to : " + str(output_file))

        properties = {
            "rows": len(df),
            "columns": len(df.columns),
            "data": output_file
        }

    except Exception:
        input_container.logger.exception ("Exception raised, stopping job execution.")
    else:
        output_node = Node("ResultSegment", f"slice-{input_container._container_id}", properties)
        output_container.put(output_node)
        

if __name__ == "__main__":
    cli()
