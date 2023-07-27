# General imports
from pathlib import Path
from typing import List, Union

import fire
import fsspec
import numpy as np
import pandas as pd
import tifffile
import tiffslide
from fsspec import open
from loguru import logger
from multimethod import multimethod

from luna.common.models import TileSchema
from luna.common.utils import get_config, save_metadata, timed


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tiles_urlpath: str = "",
    label_cols: List[str] = "???",  # type: ignore
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Converts categorical tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): url/path to valid SlideTiles table
        label_cols (List[str]): list of label columns in the input_slide_tiles table to generate the mask with
        output_urlpath (str): output url/path prefix
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        dict: output properties

    """
    config = get_config(vars())

    logger.info("Reading SlideTiles")
    with open(config["tiles_urlpath"], "rb", **config["storage_options"]) as of:
        tiles_df = pd.read_parquet(of).reset_index().set_index("address")

    with open(config["slide_urlpath"], **config["storage_options"]) as of:
        slide = tiffslide.TiffSlide(of)
        slide_width = slide.dimensions[0]
        slide_height = slide.dimensions[1]

    mask_arr, mask_values = convert_tiles_to_mask(
        tiles_df,
        slide_width,
        slide_height,
        config["label_cols"],
    )

    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    slide_mask = Path(output_urlpath_prefix) / "tile_mask.tif"
    logger.info(f"Saving output mask to {slide_mask}")
    with fs.open(slide_mask, "wb") as of:
        tifffile.imwrite(of, mask_arr)

    properties = {
        "slide_mask": slide_mask,
        "mask_values": mask_values,
        "mask_size": mask_arr.shape,
    }
    logger.info(properties)
    return properties


@multimethod
def convert_tiles_to_mask(
    tiles_df: pd.DataFrame,
    slide: tiffslide.TiffSlide,
    label_cols: Union[str, List[str]],
):
    """Converts categorical tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        tiles_df (pd.DataFrame): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        slide (tiffslide.TiffSlide): slide object
        label_cols (Union[str, List[str]]): column with labels or list of label columns in the tiles_urlpath table to generate the mask with

    Returns:
        np.ndarray, Dict[int, str]: image mask, mask value mapping

    """

    slide_width = slide.dimensions[0]
    slide_height = slide.dimensions[1]
    return convert_tiles_to_mask(tiles_df, slide_width, slide_height, label_cols)


@multimethod
def convert_tiles_to_mask(
    tiles_df: pd.DataFrame,
    slide_width: int,
    slide_height: int,
    label_cols: Union[str, List[str]],
):
    """Converts categorical tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        tiles_df (pd.DataFrame): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        slide_width (int): slide width
        slide_height (int): slide height
        label_cols (Union[str, List[str]]): column with labels or list of label columns in the tiles_urlpath table to generate the mask with

    Returns:
        np.ndarray, Dict[int, str]: image mask, mask value mapping

    """

    TileSchema.validate(tiles_df.reset_index())

    mask_arr = np.zeros((slide_height, slide_width), dtype=np.int8)

    if type(label_cols) == str:
        uniques = tiles_df[label_cols].unique()
        tiles_df["mask"] = tiles_df[label_cols].astype("category")
        mask_values = {k: v + 1 for v, k in enumerate(uniques)}
    else:
        tiles_df["mask"] = tiles_df[label_cols].idxmax(axis=1)
        tiles_df["mask"] = tiles_df["mask"].astype("category")
    mask_values = dict(zip(tiles_df["mask"], tiles_df["mask"].cat.codes + 1))

    logger.info(f"Mapping label column to mask values: {mask_values}")

    for address, row in tiles_df.iterrows():
        x, y, extent = int(row.x_coord), int(row.y_coord), int(row.xy_extent)

        value = mask_values[row["mask"]]

        # permuted rows and columns due to differences in indexing between openslide and skimage/numpy
        mask_arr[y : y + extent, x : x + extent] = value

        logger.info(f"{address}, {row['mask']}, {value}")

    return mask_arr, mask_values


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
