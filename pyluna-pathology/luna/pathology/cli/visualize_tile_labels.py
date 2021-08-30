
# General imports
import os, json, logging, pathlib
import click
import tempfile
import subprocess
import yaml

# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore_v2
from luna.common.config          import ConfigSet

from luna.pathology.common.preprocess import create_tile_thumbnail_image


@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with parameters for creating a heatmap and optionally pushing the annotation to DSA.')
def cli(app_config, datastore_id, method_param_path):
    """Visualize tile scores from inference.

    app_config - application configuration yaml file. See config.yaml.template for details.

    datastore_id - datastore name. usually a slide id.

    method_param_path - json file with parameters for creating a heatmap and optionally pushing the annotation to DSA.

    - input_wsi_tag: job tag used in loading the whole slide image

    - input_label_tag: job tag used in generating tile labels

    - job_tag: job tag for this visualization

    - scale_factor: scale for generation of thumbnails, e.g. 8 will generate a thumbnail scaled at 1/8 of the wsi.

    - tile_size: requested tile size

    - requested_magnification: requested of the slide

    - dsa_config: map of DSA instance details. e.g. {
          "host": "localhost",
          "port": "8080",
          "token": "abc123"
        }

    - root_path: path to output directory
    """
    init_logger()

    with open(method_param_path, 'r') as yaml_file:
        method_data = yaml.safe_load(yaml_file)
    visualize_tile_labels_with_datastore(app_config, datastore_id, method_data)

def visualize_tile_labels_with_datastore(app_config: str, datastore_id: str, method_data: dict):
    """Visualize tile scores from inference.

    Args:
        app_config (string): path to application configuration file.
        datastore_id (string): datastore name. usually a slide id.
        method_data (dict): method parameters including input, output details.

    Returns:
        None
    """
    logger = logging.getLogger(f"[datastore={datastore_id}]")

    # Do some setup
    cfg = ConfigSet("APP_CFG",  config_file=app_config)
    datastore   = DataStore_v2(method_data['root_path'])
    method_id   = method_data.get("job_tag", "none")

    # get slide properties
    slide_path          = datastore.get(datastore_id, method_data['input_wsi_tag'], "WholeSlideImage")
    slidestore_path     = datastore.get(datastore_id, method_data['input_wsi_tag'], "WholeSlideImage", realpath=False)
    if slidestore_path is None:
        raise ValueError("Image node not found")
    slide_metadata_json = os.path.join(pathlib.Path(slidestore_path).parent, "metadata.json")

    with open(slide_metadata_json, "r") as fp:
        slide_properties = json.load(fp)
    method_data.update(slide_properties)

    label_path  = datastore.get(datastore_id, method_data['input_label_tag'], "TileScores")
    label_metadata_path = os.path.join(label_path, "metadata.json")
    label_path = os.path.join(label_path, "tile_scores_and_labels_pytorch_inference.csv")
    with open(label_metadata_path, "r") as fp:
        label_properties = json.load(fp)

    try:

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(method_data.get("root_path"), datastore_id, method_id, "TileScores", "data")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = create_tile_thumbnail_image(slide_path, label_path, output_dir, method_data)

        # push results to DSA
        if method_data.get("dsa_config", None):
            properties = label_properties

            properties["column"]   = "tumor_score"
            properties["input"]    = label_properties["data"]
            properties["annotation_name"]   = method_id
            properties["tile_size"]   = method_data["tile_size"]
            # inference result doesn't need to be scaled. set to 1
            properties["scale_factor"]   = 1
            properties["requested_magnification"]   = method_data["requested_magnification"]
            properties["output_folder"]   = method_data["output_folder"]
            properties["image_filename"] = datastore_id + ".svs"
            with tempfile.TemporaryDirectory() as tmpdir:
                print (tmpdir)
                with open(f"{tmpdir}/model_inference_config.json", "w") as f:
                    json.dump(properties, f)
                with open(f"{tmpdir}/dsa_config.json", "w") as f:
                    json.dump(method_data["dsa_config"], f)

                # build viz
                result = subprocess.run(["python3","-m","luna.pathology.cli.dsa.dsa_viz",
                                         "-s", "heatmap",
                                         "-d", f"{tmpdir}/model_inference_config.json"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                print(result.returncode, result.stdout, result.stderr)

                # push results to DSA
                result_path = result.stdout.split(" ")[-1].strip()
                properties["annotation_filepath"] = result_path
                properties["collection_name"] = method_data["collection_name"]
                with open(f"{tmpdir}/model_inference_config.json", "w") as f:
                    json.dump(properties, f)

                subprocess.run(["python3","-m","luna.pathology.cli.dsa.dsa_upload",
                                 "-c", f"{tmpdir}/dsa_config.json", "-d", f"{tmpdir}/model_inference_config.json"])

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(properties, fp)



if __name__ == "__main__":
    cli()
