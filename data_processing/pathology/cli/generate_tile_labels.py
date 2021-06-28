'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a slide (container) ID
1. resolve the path to the WSI image
2. perform various scoring and labeling to tiles
3. save tiles as a csv with schema [address, coordinates, *scores, *labels ]

Example:
python3 -m data_processing.pathology.cli.generate_tile_labels \
    -c TCGA-BRCA \
    -s tcga-gm-a2db-01z-00-dx1.9ee36aa6-2594-44c7-b05c-91a0aec7e511 \
    -m data_processing/pathology/cli/examples/generate_tile_labels.json 

Example with annotation:
python3 -m data_processing.pathology.cli.generate_tile_labels \
    -c TCGA-BRCA \
    -s tcga-gm-a2db-01z-00-dx1.9ee36aa6-2594-44c7-b05c-91a0aec7e511 \
    -m data_processing/pathology/cli/examples/generate_tile_labels_with_ov_labels.json 
'''

# General imports
import os, json, logging
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

# From radiology.common
from data_processing.pathology.common.preprocess   import pretile_scoring

@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-c', '--cohort_id', required=True,
              help="cohort name")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with method parameters for tile generation and filtering.')
def cli(app_config, cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    generate_tile_labels_with_datastore(app_config, cohort_id, datastore_id, method_data)

def generate_tile_labels_with_datastore(app_config: str, cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, score and generate tile addresses
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    # Do some setup
    cfg = ConfigSet("APP_CFG", config_file=app_config)
    datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
    method_id   = method_data.get("job_tag", "none")

    image_node  = datastore.get("WholeSlideImage", method_data['input_wsi_tag'])
    
    # get image_id
    # TODO - allow -s to take in slide (container) id
    image_id = datastore._name

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(method_data.get("root_path"),
                                  datastore._namespace_id, datastore._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        print("Writing to output dir:", output_dir)
        properties = pretile_scoring(image_node.data, output_dir, method_data.get("root_path"), method_data, image_id)

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    # Put results in the data store
    output_node = Node("TileImages", method_id, properties)
    datastore.put(output_node)
        


if __name__ == "__main__":
    cli()
