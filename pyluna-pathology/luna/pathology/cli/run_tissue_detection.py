# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('detect_tissue')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('input_slide_tiles', str), ('requested_magnification', float), ('filter_query', str), ('output_dir', str), ('num_cores', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-rmg', '--requested_magnification', required=False,
              help="Magnificiation scale at which to perform tissue detection")
@click.option('-fq', '--filter_query', required=False,
              help="Filter query (pandas format)")          
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Run a model with a specific pre-transform for all tiles in a slide (tile_images)

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles: slide tiles (manifest tile files, .tiles.csv)
    \b
    Outputs:
        slide_tiles
    \b
    Example:
        run_tissue_detection 10001.svs 10001/tiles
            -rmg 0.5 -nc 8
            -rq 'otsu_score > 0.1 or stain0_score > 0.1'
            -o 10001/filtered_tiles
    """
    cli_runner( cli_kwargs, _params_, detect_tissue)


import pandas as pd
from tqdm import tqdm

import openslide
from luna.pathology.common.utils import get_downscaled_thumbnail, get_scale_factor_at_magnfication, \
    get_stain_vectors_macenko, pull_stain_channel, get_tile_from_slide

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from pathlib import Path
from PIL import Image, ImageEnhance



def compute_otsu_score(iterrow: tuple, slide, otsu_threshold: float) -> float:
    """
    Return otsu score for the tile.

    Args:
        iterrow (pd.Series): row with tile metadata
        slide (str): path to slide
        otsu_threshold (float): otsu threshold value
    """
    index, row = iterrow

    tile = get_tile_from_slide(row, slide, size=(10, 10))

    score = np.mean(rgb2gray(tile) < otsu_threshold)

    return score

def compute_purple_score(iterrow: tuple, slide) -> float:
    """
    Return purple score for the tile.

    Args:
        iterrow (pd.Series): row with tile metadata
        slide (str): path to slide
    """
    index, row = iterrow

    tile = get_tile_from_slide(row, slide, size=(10, 10))

    r, g, b = tile[..., 0], tile[..., 1], tile[..., 2]
    score = np.mean((r > (g + 10)) & (b > (g + 10)))
    return score

def compute_stain_score(iterrow: pd.DataFrame, slide, vectors, channel, stain_threshold:float) -> float:
    """
    Returns stain score for the tile

    Args:
        iterrow (pd.Series): row with tile metadata
        slide (str): path to slide
        vectors (np.ndarray): stain vectors
        channel (int): stain channel
        stain_threshold (float): stain threshold value
    """
    index, row = iterrow

    tile = get_tile_from_slide(row, slide, size=(10, 10))

    stain = pull_stain_channel(tile, vectors=vectors, channel=channel)
    score = np.mean (stain > stain_threshold)
    return score

def detect_tissue(input_slide_image, input_slide_tiles, requested_magnification, filter_query, output_dir, num_cores):
    """Run simple/deterministic tissue detection algorithms based on a filter query, to reduce tiles to those (likely) to contain actual tissue

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        requested_magnification (float): Magnification scale at which to perform computation
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        output_dir (str): output/working directory
        num_cores (int): Number of cores to use for CPU parallelization

    Returns:
        dict: metadata about function call
    """
    slide = openslide.OpenSlide(input_slide_image)
    slide_id = Path(input_slide_image).stem
    df = pd.read_parquet(input_slide_tiles).reset_index().set_index('address')

    logger.info (f"Slide dimensions {slide.dimensions}")

    to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    logger.info (f"Thumbnail scale factor: {to_mag_scale_factor}")

    # Origonal thumbnail
    sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)
    logger.info (f"Sample array size: {sample_arr.shape}")
    Image.fromarray(sample_arr).save(output_dir + '/sample_arr.png')

    # Enhance to drive stain apart from shadows
    enhanced_sample_img = ImageEnhance.Contrast(ImageEnhance.Color(Image.fromarray(sample_arr)).enhance(10)).enhance(10) # This pushes darks from colors
    enhanced_sample_img.save(output_dir + '/enhanced_sample_arr.png')

    # Look at HSV space
    hsv_sample_arr = np.array(enhanced_sample_img.convert('HSV'))
    Image.fromarray(np.array(hsv_sample_arr)).save(output_dir + '/hsv_sample_arr.png')

    # Look at max of saturation and value
    hsv_max_sample_arr = np.max(hsv_sample_arr[:, :, 1:3], axis=2)
    Image.fromarray(hsv_max_sample_arr).save(output_dir + '/hsv_max_sample_arr.png')

    # Get shadow mask and filter it out before other estimations
    shadow_mask = np.where (np.max(hsv_sample_arr, axis=2) < 10, 255, 0).astype(np.uint8)
    Image.fromarray(shadow_mask).save(output_dir + '/shadow_mask.png')

    # Filter out shadow/dust/etc
    sample_arr_filtered = np.where (np.expand_dims(shadow_mask, 2)==0,  sample_arr, np.full(sample_arr.shape, 255)).astype(np.uint8)
    Image.fromarray(sample_arr_filtered).save(output_dir + '/sample_arr_filtered.png')

    # Get otsu threshold
    threshold = threshold_otsu(rgb2gray(sample_arr_filtered))

    # Get stain vectors
    stain_vectors = get_stain_vectors_macenko(sample_arr_filtered)

    # Get stain thumnail image
    deconv_sample_arr = pull_stain_channel(sample_arr_filtered, vectors=stain_vectors)
    Image.fromarray(deconv_sample_arr).save(output_dir + '/deconv_sample_arr.png')

    # Get stain background thresholds
    threshold_stain0 = threshold_otsu(pull_stain_channel(sample_arr_filtered, vectors=stain_vectors, channel=0).flatten())
    threshold_stain1 = threshold_otsu(pull_stain_channel(sample_arr_filtered, vectors=stain_vectors, channel=1).flatten())

    # Get the otsu mask
    otsu_mask = np.where(rgb2gray(sample_arr_filtered) < threshold, 255, 0).astype(np.uint8)
    Image.fromarray(otsu_mask).save(output_dir + '/otsu_mask.png')

    # Get the stain masks
    stain0_mask = np.where(deconv_sample_arr[..., 0] > threshold_stain0, 255, 0).astype(np.uint8)
    stain1_mask = np.where(deconv_sample_arr[..., 1] > threshold_stain1, 255, 0).astype(np.uint8)
    Image.fromarray(stain0_mask).save(output_dir + '/stain0_mask.png')
    Image.fromarray(stain1_mask).save(output_dir + '/stain1_mask.png')

    # Be smart about computation time
    with ThreadPoolExecutor(num_cores) as p:
        if 'otsu_score' in filter_query:
            logger.info(f"Starting otsu thresholding, threshold={threshold}")
            df['otsu_score']   = list(tqdm(p.map(partial(compute_otsu_score, slide=slide, otsu_threshold=threshold), df.iterrows()), total=len(df)))
        if 'purple_score' in filter_query:
            logger.info(f"Starting purple scoring")
            df['purple_score'] = list(tqdm(p.map(partial(compute_purple_score, slide=slide), df.iterrows()), total=len(df)))
        if 'stain0_score' in filter_query:
            logger.info(f"Starting stain thresholding, channel=0, threshold={threshold_stain0}")
            df['stain0_score'] = list(tqdm(p.map(partial(compute_stain_score, slide=slide, vectors=stain_vectors, channel=0, stain_threshold=threshold_stain0), df.iterrows()), total=len(df)))
        if 'stain1_score' in filter_query:
            logger.info(f"Starting stain thresholding, channel=1, threshold={threshold_stain1}")
            df['stain1_score'] = list(tqdm(p.map(partial(compute_stain_score, slide=slide, vectors=stain_vectors, channel=1, stain_threshold=threshold_stain0), df.iterrows()), total=len(df)))
        
    logger.info (f"Filtering based on query: {filter_query}")
    df = df.query(filter_query)

    logger.info (df)

    output_header_file = f"{output_dir}/{slide_id}-filtered.tiles.parquet"

    df.to_parquet(output_header_file)

    properties = {
        "slide_tiles": output_header_file,
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
