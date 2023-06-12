# General imports
from pathlib import Path
from typing import List, Optional

import fire
import fsspec
import numpy as np
import pandas as pd
import tifffile
import tiffslide
from fsspec import open
from loguru import logger 

from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.cli.generate_tiles import generate_tiles


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tiles_urlpath: str = "",
    label_cols: List[str] = "???",  # type: ignore
    tile_size: Optional[int] = None,
    requested_magnification: Optional[int] = None,
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Converts categorical tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to valid SlideTiles table
        label_cols (List[str]): list of label columns in the input_slide_tiles table to generate the mask with
        tile_size (int): tile size
        requested_magnification (int): Magnification scale at which to perform computation
        storage_options (dict): storage options to pass to reading functions

    Returns:
        np.ndarray: image mask

    """
    config = get_config(vars())
    mask_arr = convert_tiles_to_mask(
        config["slide_urlpath"],
        config["tiles_urlpath"],
        config["label_cols"],
        config["tile_size"],
        config["requested_magnification"],
        config["storage_options"],
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
        "mask_size": mask_arr.shape,
    }
    logger.info(properties)
    return properties


def convert_tiles_to_mask(
    slide_urlpath: str,
    tiles_urlpath: str,
    label_cols: List[str],
    tile_size: Optional[int],
    requested_magnification: Optional[int],
    storage_options: dict,
):
    """Converts categorical tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to valid SlideTiles table
        label_cols (List[str]): list of label columns in the input_slide_tiles table to generate the mask with
        tile_size (int): tile size
        requested_magnification (int): Magnification scale at which to perform computation
        storage_options (dict): storage options to pass to reading functions

    Returns:
        np.ndarray: image mask

    """

    with open(slide_urlpath, **storage_options) as of:
        slide = tiffslide.TiffSlide(of)
        w = slide.dimensions[0]
        h = slide.dimensions[1]

    wsi_shape = h, w  # open slide has reversed conventions
    logger.info(f"Slide shape={wsi_shape}")

    if tiles_urlpath:
        logger.info("Reading SlideTiles")
        with open(tiles_urlpath, "rb", **storage_options) as of:
            tile_df = pd.read_parquet(of).reset_index().set_index("address")
    elif type(tile_size) == int:
        tile_df = generate_tiles(
            slide_urlpath, tile_size, storage_options, requested_magnification
        )
    else:
        raise ValueError("Specify tile size or url/path to tiling data")

    if not set(label_cols).issubset(tile_df.columns):
        raise ValueError(f"Invalid label_cols={label_cols}, verify input dataframe")

    mask_arr = np.zeros((h, w), dtype=np.int8)

    tile_df["mask"] = tile_df[label_cols].idxmax(axis=1)

    mask_values = {k: v + 1 for v, k in enumerate(label_cols)}
    logger.info(f"Mapping label column to mask values: {mask_values}")

    for address, row in tile_df.iterrows():
        x, y, extent = int(row.x_coord), int(row.y_coord), int(row.xy_extent)

        value = mask_values[row["mask"]]

        # permuted rows and columns due to differences in indexing between openslide and skimage/numpy
        mask_arr[y : y + extent, x : x + extent] = value

        logger.info(f"{address}, {row['mask']}, {value}")

    return mask_arr


if __name__ == "__main__":
    fire.Fire(cli)
