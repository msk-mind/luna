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
from data_processing.common.Container       import Container
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

from data_processing.pathology.common.preprocess import save_tiles

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = init_logger("save_tiles.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, container_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    save_tiles_with_container(cohort_id, container_id, method_data)

def save_tiles_with_container(cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
    """

    # Do some setup
    container = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)

    method_id            = method_data.get("job_tag", "none")
    output_container_id  = method_data.get("output_container")

    image_node  = container.get("WholeSlideImage", method_data['input_wsi_tag']) 
    label_node  = container.get("TileScores", method_data['input_label_tag']) 

    # Add properties to method_data
    method_data.update(label_node.properties)

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = save_tiles(image_node.get_path(), label_node.get_path(), output_dir, method_data )

    except Exception:
        container.logger.exception ("Exception raised, stopping job execution.")
    else:

        output_node = Node("TileImages", method_id, properties)
        logger.info(output_node) 
        container.add(output_node)
        container.saveAll()

    
    container = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)

    image_node  = container.get("TileImages", method_id) 

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        df = pd.read_csv(image_node.get_path(type="pathlib").joinpath("address.slice.csv"))
        df.loc[:,"object_bucket"] = image_node.properties['object_bucket']
        df.loc[:,"object_path"]   = image_node.properties['object_folder'] + "/tiles.slice.pil"
        logger.info(df)

        output_container = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(output_container_id)

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
        output_node = Node("parquet", f"slice-{container._container_id}", properties)
        output_container.add(output_node)
        output_container.saveAll()





if __name__ == "__main__":
    cli()
