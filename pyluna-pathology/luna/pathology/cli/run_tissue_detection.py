# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_tiles')

from luna.common.utils import cli_runner

import pandas as pd
from tqdm import tqdm

import openslide
from luna.pathology.common.ml import BaseTorchTileDataset


_params_ = [('input_slide_image', str), ('input_slide_tiles', str), ('output_dir', str), ('num_cores', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Run a model with a specific pre-transform for all tiles in a slide (tile_images)

    \b
    Inputs:
        input_slide_image: path to slide tiles (.svs)
    \b
    Outputs:
        slide_tiles
    \b
    Example:
        infer_tiles tiles/slide-100012/tiles
    """
    cli_runner( cli_kwargs, _params_, detect_tissue)

from luna.pathology.common.preprocess import get_downscaled_thumbnail, get_full_resolution_generator, get_scale_factor_at_magnfication
from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np
from multiprocessing import Pool
from functools import partial

from pathlib import Path
from PIL import Image

def read_tile_bytes(row):
    with open(row.tile_image_binary, "rb") as fp:
                fp.seek(int(row.tile_image_offset))
                img = Image.frombytes(
                    row.tile_image_mode,
                    (int(row.tile_image_size_xy), int(row.tile_image_size_xy)),
                    fp.read(int(row.tile_image_length)),
                )    
    return row.name, img

def compute_otsu_score(row, otsu_threshold):
    _, tile = read_tile_bytes(row)
    score = np.mean(rgb2gray(np.array(tile)) < otsu_threshold)
    return score

def detect_tissue(input_slide_image, input_slide_tiles, output_dir, num_cores):
    """Generate tile addresses, scores and optionally annotation labels using models stored in torch.hub format

    Args:
        input_slide_image: slide image (.svs)
        output_dir (str): output/working directory
        num_cores (int): how many cores to use for dataloading

    Returns:
        dict: metadata about function call
    """
    slide = openslide.OpenSlide(input_slide_image)

    logger.info (f"Slide dimensions {slide.dimensions}")

    to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=0.5)

    sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)

    threshold = threshold_otsu(rgb2gray(sample_arr))

    df = pd.read_csv(input_slide_tiles)

    logger.info(f"Starting otsu thresholding, threshold={threshold}")
    with Pool(num_cores) as p:
        df['otsu_score'] = p.map(partial(compute_otsu_score, otsu_threshold=threshold), [row for _, row in df.iterrows()])

    df = df.loc[df['otsu_score'] > 0.2]
    
    logger.info (df)

    slide_name = Path(input_slide_image).stem
    output_header_file = f"{output_dir}/{slide_name}-otsu_detections={threshold}.tiles.pil"

    df.to_csv(output_header_file)

    properties = {
        "slide_tiles": output_header_file,
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
