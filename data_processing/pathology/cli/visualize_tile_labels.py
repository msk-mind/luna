'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a slide (container) ID
1. resolve the path to the WSI image
2. perform various scoring and labeling to tiles
3. save tiles as a csv with schema [address, coordinates, *scores, *labels ]

Example:
python3 -m data_processing.pathology.cli.visualize_tile_labels \
    -c TCGA-BRCA \
    -s tcga-gm-a2db-01z-00-dx1.9ee36aa6-2594-44c7-b05c-91a0aec7e511 \
    -m data_processing/pathology/cli/example_visualize_tile_labels.json

Example with annotation:
python3 -m data_processing.pathology.cli.visualize_tile_labels \
        -c ov-path-druv  \
        -s 226871 \
        -m data_processing/pathology/cli/example_visualize_tile_labels.json 
'''

# General imports
import os, json, sys
import click
import tempfile
import subprocess

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.utils           import get_method_data
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

# From radiology.common
from data_processing.pathology.common.preprocess   import visualize_scoring

logger = init_logger("visualize_tile_labels.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, container_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    visualize_tile_labels_with_container(cohort_id, container_id, method_data)

def visualize_tile_labels_with_container(cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
    """

    # Do some setup
    container   = DataStore( cfg ).setNamespace(cohort_id).setContainer(container_id)
    method_id   = method_data.get("job_tag", "none")
    
    image_node  = container.get("WholeSlideImage", method_data['input_wsi_tag']) 
    label_node  = container.get("TileScores",      method_data['input_label_tag']) 

    method_data.update(label_node.properties)

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # properties = visualize_scoring(image_node.data, label_node.data, output_dir, method_data)
        if method_data.get("dsa_config", None):
            params = label_node.properties

            params["column"]            = "tumor_score"
            params["input"]             = label_node.properties["data"]
            params["output_folder"]     = output_dir
            params["output"]   = method_id

            with tempfile.TemporaryDirectory() as tmpdir:
                print (tmpdir)
                with open(f"{tmpdir}/model_inference_config.json", "w") as f: json.dump(params, f)
                with open(f"{tmpdir}/dsa_config.json", "w") as f: json.dump(method_data["dsa_config"], f)
                subprocess.call(["dsa", "-c", f"{tmpdir}/dsa_config.json", "heatmap", "-d", f"{tmpdir}/model_inference_config.json"])

                


    except Exception:
        container.logger.exception ("Exception raised, stopping job execution.")
    else:
        output_node = Node("TileScores", method_id, properties)
        container.add(output_node)
        container.saveAll()


if __name__ == "__main__":
    cli()
