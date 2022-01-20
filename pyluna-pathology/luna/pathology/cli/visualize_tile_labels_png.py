# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger  import init_logger

init_logger()
logger = logging.getLogger('visualize_tiles_png')

from luna.common.utils import cli_runner

import pandas as pd
from tqdm import tqdm

from typing import List

_params_ = [('input_slide_image', str), ('input_slide_tiles', str), ('plot_labels', List[str]), ('requested_magnification', float), ('output_dir', str)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-pl', '--plot_labels', required=False,
              help='labels_to_plot')
@click.option('-rmg', '--requested_magnification', required=False,
              help="Magnificiation scale at which to generate thumbnail/png images (recommended <= 1)")
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Generate nice tile markup images with continuous or discrete tile scores

    \b
    Inputs:
        input_slide_image: path to slide image (.svs)
        input_slide_tiles: path to tile images (.tiles.csv)
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

from luna.pathology.common.preprocess import get_downscaled_thumbnail, get_scale_factor_at_magnfication
from luna.pathology.common.preprocess import get_tile_color
from skimage.draw import rectangle_perimeter

from pathlib import Path
from PIL import Image
import numpy as np

def visualize_tiles(input_slide_image, input_slide_tiles, requested_magnification, plot_labels, output_dir):
    """Generate nice tile markup images with continuous or discrete tile scores

    Args:
        input_slide_image (str): path to slide image (.svs)
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        requested_magnification (float): Magnification scale at which to perform computation
        plot_labels (List[str]): labels to plot
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    slide = openslide.OpenSlide(input_slide_image)
 
    to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    # Create thumbnail image for scoring
    sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)

    # Get tiles
    df = pd.read_csv(input_slide_tiles).set_index('address')

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


def visualize_tiling_scores(df:pd.DataFrame, thumbnail_img:np.ndarray, scale_factor:float,
        score_type_to_visualize:str) -> np.ndarray:
    """visualize tile scores
    
    draws colored boxes around tiles to indicate the value of the score 

    Args:
        df (pd.DataFrame): input dataframe
        thumbnail_img (np.ndarray): input tile 
        tile_size (int): tile width/length
        score_type_to_visualize (str): column name from data frame
    
    Returns:
        np.ndarray: new thumbnail image with boxes around tiles passing indicating the
        value of the score
    """

    assert isinstance(thumbnail_img, np.ndarray)

    for _, row in tqdm(df.iterrows()):

        if 'regional_label' in row and pd.isna(row.regional_label): continue

        start = (row.y_coord / scale_factor, row.x_coord / scale_factor)  # flip because OpenSlide uses (column, row), but skimage, uses (row, column)

        rr, cc = rectangle_perimeter(start=start, extent=(row.full_resolution_tile_size/ scale_factor, row.full_resolution_tile_size/ scale_factor), shape=thumbnail_img.shape)
        
        # set color based on intensity of value instead of black border (1)
        score = row[score_type_to_visualize]

        thumbnail_img[rr, cc] = get_tile_color(score)
    
    return thumbnail_img

if __name__ == "__main__":
    cli()
