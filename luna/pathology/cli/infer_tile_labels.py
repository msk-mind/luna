# General imports
import os, json, logging
import click
import yaml

# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore_v2
from luna.common.config          import ConfigSet

# From radiology.common
from luna.pathology.common.preprocess   import run_model

@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with method parameters for loading a saved model.')
def cli(app_config, datastore_id, method_param_path):
    """Infer tile labels with a trained model.

    app_config - application configuration yaml file. See config.yaml.template for details.

    datastore_id - datastore name. usually a slide id.

    method_param_path - json file with method parameters for loading a saved model.

    - input_label_tag: job tag used to generate tile labels

    - job_tag: job tag for the inference

    - model_package: package to load your model e.g. luna.pathology.models.ov_tissuenet

    - model: model details e.g. {
          "checkpoint_path": "/path/to/checkpoint",
          "n_classes": 4
      }

    - root_path: path to output directory
    """
    init_logger()

    with open(method_param_path, 'r') as yaml_file:
        method_data = yaml.safe_load(yaml_file)
    infer_tile_labels_with_datastore(app_config, datastore_id, method_data)

def infer_tile_labels_with_datastore(app_config: str, datastore_id: str, method_data: dict):
    """Infer tile labels with a trained model.

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
    datastore   = DataStore_v2(method_data.get("root_path"))
    method_id   = method_data.get("job_tag", "none")

    tile_path           = datastore.get(datastore_id, method_data['input_label_tag'], "TileImages")
    if tile_path is None:
        raise ValueError("Tile path not found")

    tile_image_path     = os.path.join(tile_path, "tiles.slice.pil")
    tile_label_path     = os.path.join(tile_path, "address.slice.csv")

    # get image_id
    # TODO - allow -s to take in slide (container) id

    try:
        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(method_data.get("root_path"), datastore_id, method_id, "TileScores", "data")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        properties = run_model(tile_image_path, tile_label_path, output_dir, method_data)

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(properties, fp)


if __name__ == "__main__":
    cli()
