# General imports
import logging
from typing import List
import click
import numpy as np
import openslide
import pandas as pd
import tifffile

from luna.pathology.common.schemas import SlideTiles
from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

init_logger()
logger = logging.getLogger("convert_tiles_to_mask")

_params_ = [
    ("input_slide_image", str),
    ("input_slide_tiles", str),
    ("output_dir", str),
    ("label_cols", List[str]),
]


@click.command()
@click.argument("input_slide_image", nargs=1)
@click.argument("input_slide_tiles", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-lc",
    "--label_cols",
    required=False,
    help="columns whose values are used to generate the mask",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """Generates a .tif mask from slide tile labels

    \b
    Inputs:
        input: input data
    \b
    Outputs:
        output data
    \b
    Example:
        convert_tiles_to_mask ./slides/10001.svs ./tiles_scores_and_labels.parquet
            -lc Background,Tumor
            -o ./label_mask.parquet
    """
    cli_runner(cli_kwargs, _params_, convert_tiles_to_mask)


def convert_tiles_to_mask(
    input_slide_image: str,
    input_slide_tiles: str,
    label_cols: List[str],
    output_dir: str,
):
    """Converts cateogrial tile labels to a slide image mask. This mask can be used for feature extraction and spatial analysis.

     Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to valid SlideTiles table
        label_cols (List[str]): list of label columns in the input_slide_tiles table to generate the mask with
        output_dir (str): output/working directory

    Returns:
        dict: output .tif path and the mask size

    """

    slide = openslide.OpenSlide(input_slide_image)

    w = slide.dimensions[0]
    h = slide.dimensions[1]
    wsi_shape = h, w  # open slide has reversed conventions
    logger.info(f"Slide shape={wsi_shape}")

    logger.info("Validating SlideTiles schema")
    assert SlideTiles.check(f"{input_slide_tiles}")  # verfiying SlideTiles schema

    # check if tile_col is a valid argument
    logger.info("Reading SlideTiles")
    print(label_cols)
    tile_df = pd.read_parquet(input_slide_tiles).reset_index().set_index("address")

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

    slide_mask = f"{output_dir}/tile_mask.tif"
    logger.info(f"Saving output mask to {slide_mask}")
    tifffile.imsave(slide_mask, mask_arr)

    properties = {
        "slide_mask": slide_mask,
        "mask_size": list(wsi_shape),
    }
    logger.info(properties)
    return properties


if __name__ == "__main__":
    cli()
