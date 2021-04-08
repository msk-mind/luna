'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to a volumentric image and annotation (label) files
2. resample image and segmentation, and save voxels as a 3d numpy array (.npy)
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

# From radiology.common
from data_processing.radiology.common.preprocess   import randomize_contours

logger = init_logger("randomize_contours.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, container_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    randomize_contours_with_container(cohort_id, container_id, method_data)

def randomize_contours_with_container(cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, perform MIRP contour randomization
    """

    # Do some setup
    container   = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_id   = method_data.get("job_tag", "none")

    image_node  = container.get("VolumetricImage", method_data['image_input_tag']) 
    label_node  = container.get("VolumetricLabel", method_data['label_input_tag'])

    try:
        if image_node is None:
            raise ValueError("Image node not found")

        if label_node is None:
            raise ValueError("Label node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", container._namespace_id, container._name, method_id)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        image_properties, label_properties, pertubation_properties, supervoxel_properties = randomize_contours(
            image_path = image_node.data,
            label_path = label_node.data,
            output_dir = output_dir,
            params     = method_data
        )

    except Exception:
        container.logger.exception ("Exception raised, stopping job execution.")
    else:
        new_image_node          = Node("VolumetricImage",    method_id, image_properties)
        new_label_node          = Node("VolumetricLabel",    method_id, label_properties)
        new_pertubation_node    = Node("VolumetricLabelSet", method_id, pertubation_properties)
        new_supervoxel_node     = Node("Voxels", method_id, supervoxel_properties)

        container.add(new_image_node)
        container.add(new_label_node)
        container.add(new_pertubation_node)
        container.add(new_supervoxel_node)
        container.saveAll()


if __name__ == "__main__":
    cli()
