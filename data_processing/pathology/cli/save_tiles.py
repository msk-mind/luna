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
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

from data_processing.pathology.common.preprocess import save_tiles

logger = init_logger("save_tiles.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    save_tiles_with_container(cohort_id, datastore_id, method_data)

def save_tiles_with_container(cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
    """

    # Do some setup
    datastore = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)

    method_id  = method_data.get("job_tag", "none")

    image_node  = datastore.get("WholeSlideImage", method_data['input_wsi_tag'])
    label_node  = datastore.get("TileScores", method_data['input_label_tag'])

    # Add properties to method_data
    method_data.update(label_node.properties)

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  datastore._namespace_id, datastore._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = save_tiles(image_node.data, label_node.data, output_dir, method_data )

    except Exception:
        datastore.logger.exception ("Exception raised, stopping job execution.")
        return

    # Put results in the data store
    output_node = Node("TileImages", method_id, properties)
    datastore.put(output_node)



if __name__ == "__main__":
    cli()
