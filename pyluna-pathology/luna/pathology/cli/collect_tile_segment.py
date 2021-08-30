# General imports
import os, json, logging, pathlib
import click
import yaml

# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore_v2
from luna.common.config          import ConfigSet

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with method parameters including input, output details.')
def cli(app_config, datastore_id, method_param_path):
    """Save tiles as a parquet file, indexed by slide id, address, and optionally patient_id.

    app_config - application configuration yaml file. See config.yaml.template for details.

    datastore_id - datastore name. usually a slide id.

    method_param_path - json file with method parameters including input, output details.

    - input_label_tag: job tag used for generating tile labels

    - input_wsi_tag: job tag used for loading the slide

    - output_datastore: job tag for collecting tiles

    - root_path: path to output data
  """
    init_logger()

    with open(method_param_path, 'r') as yaml_file:
        method_data = yaml.safe_load(yaml_file)
    collect_tile_with_datastore(app_config, datastore_id, method_data)

def collect_tile_with_datastore(app_config: str, datastore_id: str, method_data: dict):
    """Save tiles as a parquet file.

    Save tiles as a parquet file, indexed by slide id, address, and optionally patient_id.

    Args:
        app_config (string): path to application configuration file.
        datastore_id (string): datastore name. usually a slide id.
        method_data (dict): method parameters including input, output details.

    Returns:
        None
    """
    logger = logging.getLogger(f"[datastore={datastore_id}]")

    cfg = ConfigSet("APP_CFG", config_file=app_config)

    input_tile_data_id   = method_data.get("input_label_tag")
    input_wsi_tag  = method_data.get("input_wsi_tag")
    output_datastore_id  = method_data.get("output_datastore")

    # get info from WholeSlideImages and TileImages
    datastore = DataStore_v2(method_data.get("root_path"))
    slide_path          = datastore.get(datastore_id, input_wsi_tag, "WholeSlideImage", realpath=False)
    if slide_path is None:
        raise ValueError("Image node not found")
    slide_metadata_json    = os.path.join(pathlib.Path(slide_path).parent, "metadata.json")

    tile_path           = datastore.get(datastore_id, input_tile_data_id, "TileImages")
    tile_image_path     = os.path.join(tile_path, "tiles.slice.pil")
    tile_label_path     = os.path.join(tile_path, "address.slice.csv")
    tile_label_metadata_json = os.path.join(tile_path, "metadata.json")

    with open(tile_label_metadata_json, "r") as fp:
        tile_properties = json.load(fp)
    with open(slide_metadata_json, "r") as fp:
        slide_properties = json.load(fp)
    try:
        df = pd.read_csv(tile_label_path)
        df.loc[:,"data_path"]     = tile_image_path
        if cfg.get_value(path='APP_CFG::OBJECT_STORE_ENABLED'):
            df.loc[:,"object_bucket"] = tile_properties['object_bucket']
            df.loc[:,"object_path"]   = tile_properties['object_folder'] + "/tiles.slice.pil"

        if slide_path and 'patient_id' in slide_properties:
            df.loc[:,"patient_id"]   = slide_properties['patient_id']

        df.loc[:,"id_slide_container"] = datastore_id

        if 'patient_id' in df:
            df = df.set_index(["patient_id", "id_slide_container", "address"])
        else:
            df = df.set_index(["id_slide_container", "address"])
        logger.info(df)

        output_dir = os.path.join(method_data.get("root_path"), output_datastore_id)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{datastore_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_file)

        logger.info("Saved to : " + str(output_file))

        """properties = {
            "rows": len(df),
            "columns": len(df.columns),
            "data": output_file
        }
        print(properties)"""

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e



if __name__ == "__main__":
    cli()
