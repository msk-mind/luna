# General imports
import json
from pathlib import Path

import fire
import fsspec
import pandas as pd
from fsspec import open
from loguru import logger
from shapely.geometry import GeometryCollection, Polygon, shape
from tqdm import tqdm

from luna.common.utils import get_config, save_metadata, timed


@timed
@save_metadata
def cli(
    annotation_urlpath: str = "???",
    tiles_urlpath: str = "???",
    slide_id: str = "???",
    output_urlpath: str = "???",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Queries the dataset at input_slide_annotation_dataset for a slide_id matching input_slide_tiles

    Adds regional_label, intersection_area columns to slide tiles, where the former is the annotation label, and the latter the fraction of intersecting area between the tile and annotation regions

    Args:
        annotation_urlpath (str): url/path to parquet annotation dataset
        tiles_urlpath (str): url/path to a slide-tile manifest file (.tiles.parquet)
        slide_id (str): slide ID
        output_urlpath (str): output url/path prefix
        storage_options (dict): options to pass to reading functions
        output_storage_options (dict): options to pass to writing functions
        local_config (str): url/path to local config YAML file
    Returns:
        dict: metadata
    """
    config = get_config(vars())

    df_tiles = generate_tile_labels(
        config["annotation_urlpath"],
        config["tiles_urlpath"],
        config["slide_id"],
        config["storage_options"],
    )

    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_header_file = (
        Path(output_urlpath_prefix)
        / f"{config['slide_id']}.regional_label.tiles.parquet"
    )
    with fs.open(output_header_file, "wb") as of:
        df_tiles.to_parquet(of)

    properties = {
        "slide_tiles": output_header_file,  # "Tiles" are the metadata that describe them
    }

    return properties


# Transform imports
def generate_tile_labels(
    annotation_urlpath: str,
    tiles_urlpath: str,
    slide_id: str,
    storage_options: dict = {},
):
    """Queries the dataset at input_slide_annotation_dataset for a slide_id matching input_slide_tiles

    Adds regional_label, intersection_area columns to slide tiles, where the former is the annotation label, and the latter the fraction of intersecting area between the tile and annotation regions

    Args:
        annotation_urlpath (str): url/path to parquet annotation dataset
        tiles_urlpath (str): url/path to a slide-tile manifest file (.tiles.parquet)
        slide_id (str): slide ID
        storage_options (dict): options to pass to reading functions
    Returns:
        pd.DataFrame: tile dataframe with regional_label, and intersection_area columns
    """
    slide_id = str(slide_id)
    logger.info(f"slide_id={slide_id}")

    with open(annotation_urlpath, **storage_options) as of:
        df_annotation = pd.read_parquet(of)

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

    with open(tiles_urlpath, **storage_options) as of:
        df_tiles = pd.read_parquet(of).reset_index().set_index("address")
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

    return df_tiles


if __name__ == "__main__":
    fire.Fire(cli)
