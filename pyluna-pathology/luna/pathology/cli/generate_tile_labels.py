
# General imports
import os, json, logging, yaml
import click
import pandas as pd

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger() ### Add CLI tool name

from luna.common.utils import cli_runner

_params_ = [('input_slide_annotation_dataset', str), ('input_slide_tiles', str), ('output_dir', str)]

@click.command()
@click.argument('input_slide_annotation_dataset', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
### Additional options
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ A cli tool

    \b
    Inputs:
        input: input data
    \b
    Outputs:
        output data
    \b
    Example:
        CLI_TOOL ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    cli_runner( cli_kwargs, _params_, generate_tile_labels, pass_keys=True)

from shapely.geometry import shape, GeometryCollection, Polygon
from tqdm import tqdm
### Transform imports 
def generate_tile_labels(input_slide_annotation_dataset, input_slide_tiles, output_dir, keys):
    """ CLI tool method

    Args:
        input_data (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    slide_id = keys.get('slide_id', None)

    logger.info(f"slide_id={slide_id}")

    df_annotation = pd.read_parquet(input_slide_annotation_dataset).loc[slide_id].query("type=='geojson'")

    slide_geojson, collection_name, annotation_name = df_annotation.slide_geojson.item(), df_annotation.collection_name.item(), df_annotation.annotation_name.item()

    print (slide_geojson, collection_name, annotation_name)

    with open(slide_geojson) as f:
        features = json.load(f)['features']

    d_collections = {}

    for feature in features: 
        label = feature['properties']['label']

        if not label in d_collections.keys(): d_collections[label] = []

        d_collections[label].append(shape(feature["geometry"]).buffer(0))

    for label in d_collections.keys():
        d_collections[label] = GeometryCollection( d_collections[label] )

    df_tiles = pd.read_csv(input_slide_tiles).set_index('address')
    l_regional_labels = []
    l_intersection_areas = []

    for _, row in tqdm(df_tiles.iterrows(), total=len(df_tiles)):
        tile_x, tile_y, tile_extent = row.x_coord, row.y_coord, row.xy_extent

        tile_polygon = Polygon([
            (tile_x,               tile_y),
            (tile_x,               tile_y+tile_extent),
            (tile_x+tile_extent,   tile_y+tile_extent),
            (tile_x+tile_extent,   tile_y),
        ])

        tile_label = None
        max_overlap = 0.0
        for label in d_collections.keys(): 
            intersection_area = d_collections[label].intersection(tile_polygon).area / tile_polygon.area
            if intersection_area > max_overlap:
                tile_label, max_overlap = label, intersection_area

        l_regional_labels.append(tile_label)
        l_intersection_areas.append(max_overlap)

    df_tiles['regional_label']    = l_regional_labels
    df_tiles['intersection_area'] = l_intersection_areas

    print ( df_tiles.loc[df_tiles.intersection_area > 0] )

    output_header_file = f"{output_dir}/{slide_id}.regional_label.tiles.csv"
    df_tiles.to_csv(output_header_file)        
    
    properties = {
        "slide_tiles": output_header_file, # "Tiles" are the metadata that describe them
        "segment_keys": {'regional_annotation_id': f"dsa-{collection_name}-{annotation_name}"}
    }

    return properties

if __name__ == "__main__":
    cli()






































# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore_v2
from luna.common.config          import ConfigSet

# From pathology.common

@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json file with method parameters for tile generation and filtering.')
def cli(app_config, datastore_id, method_param_path):
    """Generate tile addresses, scores and optionally annotation labels.

    app_config - application configuration yaml file. See config.yaml.template for details.

    datastore_id - datastore name. usually a slide id.

    method_param_path - json file with method parameters for tile generation and filtering.

    - input_wsi_tag: job tag used to load slides

    - job_tag: job tag for generating tile labels

    - tile_size: size of patches

    - scale_factor: desired downscale factor

    - requested_magnification: desired magnification

    - root_path: path to output data

    - filter: optional filter map to select subset of the tiles e.g. {
        "otsu_score": 0.5
      }

    - project_id: optional project id, if using regional annotations

    - labelset: optional annotation labelset name, if using regional annotations

    - annotation_table_path: optional path to the regional annotation table
    """
    init_logger()

    with open(method_param_path, 'r') as yaml_file:
        method_data = yaml.safe_load(yaml_file)
    generate_tile_labels_with_datastore(app_config, datastore_id, method_data)

def generate_tile_labels_with_datastore(app_config: str, datastore_id: str, method_data: dict):
    """Generate tile addresses, scores and optionally annotation labels.

    Args:
        app_config (string): path to application configuration file.
        datastore_id (string): datastore name. usually a slide id.
        method_data (dict): method parameters including input, output details.

    Returns:
        None
    """
    logger = logging.getLogger(f"[datastore={datastore_id}]")

    # Do some setup
    cfg = ConfigSet("APP_CFG", config_file=app_config)
    datastore   = DataStore_v2(method_data.get("root_path"))
    method_id   = method_data.get("job_tag", "none")

    image_path  = datastore.get(datastore_id, method_data['input_wsi_tag'], "WholeSlideImage")
    logger.info(f"Whole slide image path: {image_path}")

    # get image_id
    # TODO - allow -s to take in slide (datastore_id) id
    image_id = datastore_id

    try:
        if image_path is None:
            raise ValueError("Image node not found")

        # Data just goes under namespace/name
        # TODO: This path is really not great, but works for now
        output_dir = os.path.join(method_data.get("root_path"), datastore_id, method_id, "TileImages", "data")
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        logger.info(f"Writing to output dir: {output_dir}")
        # properties = pretile_scoring(image_path, output_dir, method_data.get("annotation_table_path"), method_data, image_id)
        properties = {}
    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(properties, fp)

if __name__ == "__main__":
    cli()
