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
import os, json, logging
import click

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-c', '--cohort_id',    required=True,
              help="cohort name")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with method parameters including input, output details.')
def cli(app_config, cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    collect_tile_with_datastore(app_config, cohort_id, datastore_id, method_data)

def collect_tile_with_datastore(app_config: str, cohort_id: str, container_id: str, method_data: dict):
    """
    Using the container API interface, visualize tile-wise scores
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    cfg = ConfigSet("APP_CFG", config_file=app_config)

    input_tile_data_id   = method_data.get("input_label_tag")
    output_datastore_id  = method_data.get("output_datastore")

    input_datastore  = DataStore( cfg ).setNamespace(cohort_id)\
        .setDatastore(container_id)

    output_datastore = DataStore(cfg).setNamespace(cohort_id)\
        .createDatastore(output_datastore_id, "parquet")\
        .setDatastore(output_datastore_id)

    image_node  = input_datastore.get("TileImages", input_tile_data_id)

    try:
        if image_node is None:
            raise ValueError("Image node not found")
 
        df = pd.read_csv(image_node.aux)
        df.loc[:,"data_path"]     = image_node.data
        if cfg.get_value(path='APP_CFG::OBJECT_STORE_ENABLED'):
            df.loc[:,"object_bucket"] = image_node.properties['object_bucket']
            df.loc[:,"object_path"]   = image_node.properties['object_folder'] + "/tiles.slice.pil"
        df.loc[:,"id_slide_container"] = input_datastore._name

        df = df.set_index(["id_slide_container", "address"])
        logger.info(df)

        output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], method_data.get("env", "data"),
                                  output_datastore._namespace_id, output_datastore._name)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{input_datastore._datastore_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_file)

        logger.info("Saved to : " + str(output_file))

        properties = {
            "rows": len(df),
            "columns": len(df.columns),
            "data": output_file
        }
        print(properties)

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    # Put results in the data store
    output_node = Node("ResultSegment", f"slide-{input_datastore._datastore_id}", properties)
    output_datastore.put(output_node)
        

if __name__ == "__main__":
    cli()
