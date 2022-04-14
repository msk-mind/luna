# General imports
import json
import logging
import click
import pandas as pd
from shapely.geometry import shape, GeometryCollection, Polygon
from tqdm import tqdm

from luna.common.utils import cli_runner
from luna.common.custom_logger import init_logger

init_logger()
logger = logging.getLogger("generate_tile_labels")  # Add CLI tool name

_params_ = [
    ("input_slide_annotation_dataset", str),
    ("input_slide_tiles", str),
    ("output_dir", str),
]


@click.command()
@click.argument("input_slide_annotation_dataset", nargs=1)
@click.argument("input_slide_tiles", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
# Additional options
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """A cli tool

    \b
    Inputs:
        input_slide_annotation_dataset: annotation dataset containing metadata about geojsons
        input_slide_tiles: path to tile images (.tiles.parquet)
    \b
    Outputs:
        slide_tiles
    \b
    Example:
        generate_tile_labels ./annotations/ tiles/slide-100012/tiles
            -o ./labeled_tiles/10001/
    """
    cli_runner(cli_kwargs, _params_, generate_tile_labels, pass_keys=True)


# Transform imports
def generate_tile_labels(
    input_slide_annotation_dataset, input_slide_tiles, output_dir, keys
):
    """Queries the dataset at input_slide_annotation_dataset for a slide_id matching input_slide_tiles

    Adds regional_label, intersection_area columns to slide tiles, where the former is the annotation label, and the latter the fraction of intersecting area between the tile and annotation regions

    Args:
        input_slide_annotation_dataset (str): path to parquet annotation dataset
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.parquet)
        output_dir (str): output/working directory
        keys (dict): segment keys (this function needs 'slide_id')
    Returns:
        dict: metadata about function call
    """
    slide_id = keys.get("slide_id", None)

    logger.info(f"slide_id={slide_id}")

    df_annotation = pd.read_parquet(input_slide_annotation_dataset)

    if slide_id not in df_annotation.index:
        raise RuntimeError("No matching annotations found for slide!")

    df_annotation = df_annotation.loc[[slide_id]].query("type=='geojson'")

    if not len(df_annotation):
        raise RuntimeError("No matching geojson annotations found!")

    slide_geojson, collection_name, annotation_name = (
        df_annotation.slide_geojson.item(),
        df_annotation.collection_name.item(),
        df_annotation.annotation_name.item(),
    )

    print(slide_geojson, collection_name, annotation_name)

    with open(slide_geojson) as f:
        features = json.load(f)["features"]

    d_collections = {}

    for feature in features:
        label = feature["properties"]["label"]

        if label not in d_collections.keys():
            d_collections[label] = []

        d_collections[label].append(shape(feature["geometry"]).buffer(0))

    for label in d_collections.keys():
        d_collections[label] = GeometryCollection(d_collections[label])

    df_tiles = pd.read_parquet(input_slide_tiles).reset_index().set_index("address")
    l_regional_labels = []
    l_intersection_areas = []

    for _, row in tqdm(df_tiles.iterrows(), total=len(df_tiles)):
        tile_x, tile_y, tile_extent = row.x_coord, row.y_coord, row.xy_extent

        tile_polygon = Polygon(
            [
                (tile_x, tile_y),
                (tile_x, tile_y + tile_extent),
                (tile_x + tile_extent, tile_y + tile_extent),
                (tile_x + tile_extent, tile_y),
            ]
        )

        tile_label = None
        max_overlap = 0.0
        for label in d_collections.keys():
            intersection_area = (
                d_collections[label].intersection(tile_polygon).area / tile_polygon.area
            )
            if intersection_area > max_overlap:
                tile_label, max_overlap = label, intersection_area

        l_regional_labels.append(tile_label)
        l_intersection_areas.append(max_overlap)

    df_tiles["regional_label"] = l_regional_labels
    df_tiles["intersection_area"] = l_intersection_areas

    logger.info(df_tiles.loc[df_tiles.intersection_area > 0])

    output_header_file = f"{output_dir}/{slide_id}.regional_label.tiles.parquet"
    df_tiles.to_parquet(output_header_file)

    properties = {
        "slide_tiles": output_header_file,  # "Tiles" are the metadata that describe them
    }

    return properties


if __name__ == "__main__":
    cli()
