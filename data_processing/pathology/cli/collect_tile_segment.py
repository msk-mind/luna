'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a slide (container) ID
1. resolve the path to the WsiImage and TileLabels
2. perform various scoring and labeling to tiles
3. save tiles as a parquet file with schema [address, coordinates, *scores, *labels ]

Example:
python3 -m data_processing.pathology.cli.collect_tiles \
    -s tcga-gm-a2db-01z-00-dx1.9ee36aa6-2594-44c7-b05c-91a0aec7e511 \
    -m data_processing/pathology/cli/example_collect_tiles.json
'''

# General imports
import os, json, logging, pathlib
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore_v2
from data_processing.common.config          import ConfigSet

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
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    collect_tile_with_datastore(app_config, datastore_id, method_data)

def collect_tile_with_datastore(app_config: str, datastore_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
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

        output_dir = os.path.join(method_data.get("root_path"), output_datastore_id, datastore_id)

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
