# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger  import init_logger

init_logger()
logger = logging.getLogger('visualize_tiles_png')

from luna.common.utils import cli_runner

from typing import List

_params_ = [('input_slide_image', str), ('input_slide_tiles', str), ('mpp_units', bool), ('plot_labels', List[str]), ('requested_magnification', float), ('output_dir', str)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-pl', '--plot_labels', required=False,
              help='Label names (as column labels) to plot')
@click.option('-rmg', '--requested_magnification', required=False,
              help="Magnificiation scale at which to generate thumbnail/png images (recommended <= 1)")
@click.option('--mpp-units', is_flag=True,
              help="Set this flag if input coordinates are in µm, not pixels")
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Generate nice tile markup images with continuous or discrete tile scores

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles: slide tiles (manifest tile files, .tiles.csv)
    \b
    Outputs:
        markups: markup images
    \b
    Example:
        visualize_tiles_png 10001.svs 10001/tiles/10001.tiles.csv
            -o 10001/markups
            -pl Tumor,Stroma,TILs,otsu_score
            -rmg 0.5

    """
    cli_runner ( cli_kwargs, _params_, visualize_tiles)

import openslide

from luna.pathology.common.utils import visualize_tiling_scores, get_downscaled_thumbnail, get_scale_factor_at_magnfication
import pandas as pd

from pathlib import Path
from PIL import Image
def visualize_tiles(input_slide_image, input_slide_tiles, requested_magnification, mpp_units, plot_labels, output_dir):
    """Generate nice tile markup images with continuous or discrete tile scores

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        requested_magnification (float): Magnification scale at which to perform computation
        plot_labels (List[str]): labels to plot
        output_dir (str): output/working directory
        mpp_units (bool): if true, additional rescaling is applied to match micro-meter and pixel coordinate systems

    Returns:
        dict: metadata about function call
    """
    slide = openslide.OpenSlide(input_slide_image)

    
    to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    # Create thumbnail image for scoring
    sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)

    # See if we need to adjust scale_factor to account for different units
    if mpp_units: 
        unit_sf = float(slide.properties['openslide.mpp-x'])
        to_mag_scale_factor *= unit_sf

    # Get tiles
    df = pd.read_parquet(input_slide_tiles).reset_index().set_index('address')

    # only visualize tile scores that were able to be computed
    all_score_types = set(plot_labels)
    score_types_to_visualize = set(list(df.columns)).intersection(all_score_types)

    for score_type_to_visualize in score_types_to_visualize:
        output_file = os.path.join(output_dir, "tile_scores_and_labels_visualization_{}.png".format(score_type_to_visualize))

        thumbnail_overlayed = visualize_tiling_scores(df, sample_arr, to_mag_scale_factor, score_type_to_visualize)
        thumbnail_overlayed = Image.fromarray(thumbnail_overlayed)
        thumbnail_overlayed.save(output_file)

        logger.info ("Saved %s visualization at %s", score_type_to_visualize, output_file)

    properties = {'data': output_dir}

    return properties

if __name__ == "__main__":
    cli()
