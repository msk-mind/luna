# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_tiles')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('input_slide_tiles', str), ('output_dir', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-bx', '--batch_size', required=False,
              help="Batch size used for inference speedup", default=64)    
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Rasterize a slide into smaller tiles
    
    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    Outputs:
        slide_tiles
    \b
    Example:
        save_tiles 10001.svs
            -nc 8 -rts 244 -rmg 10 -bx 200
            -o 10001/tiles
    """
    cli_runner( cli_kwargs, _params_, save_tiles)

import pandas as pd
from tqdm import tqdm
import openslide
import itertools
from pathlib import Path
import h5py

from concurrent.futures import ProcessPoolExecutor, as_completed

from luna.pathology.common.utils import get_tile_from_slide
from luna.common.utils import grouper

def get_tile_array(iterrows: pd.DataFrame, input_slide_image):
    """
    Returns stain score for the tile

    Args:
        row (pd.DataFrame): row with address and tile_image_file columns
        vectors (np.ndarray): stain vectors
        channel (int): stain channel
        stain_threshold (float): stain threshold value
    """
    slide = openslide.OpenSlide(str(input_slide_image))
    return [(index, get_tile_from_slide(row, slide)) for index, row in iterrows]

def save_tiles(input_slide_image, input_slide_tiles, output_dir, num_cores, batch_size):
    """Rasterize a slide into smaller tiles
    
    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tile_size (int): size of tiles to use (at the requested magnification)
        num_cores (int): Number of cores to use for CPU parallelization
        requested_magnification (float): Magnification scale at which to perform computation
        output_dir (str): output/working directory
        batch_size (int): size in batch dimension to chuck jobs

    Returns:
        dict: metadata about function call
    """
    slide_id = Path(input_slide_image).stem
    df = pd.read_csv(input_slide_tiles).set_index('address')

    output_header_file = f"{output_dir}/{slide_id}.tiles.csv"
    output_hdf_file    = f"{output_dir}/{slide_id}.tiles.h5"

    logger.info(f"Now generating tiles with num_cores={num_cores} and batch_size={batch_size}!")
    if os.path.exists(output_hdf_file):
        logger.warning(f"{output_hdf_file} already exists, deleting the file..")
        os.remove(output_hdf_file)

    # save address:tile arrays key:value pair in hdf5
    hfile = h5py.File(output_hdf_file, 'a')
    with ProcessPoolExecutor(num_cores) as executor:
        out = [executor.submit(get_tile_array, tile_row, input_slide_image)
               for tile_row in grouper(df.iterrows(), batch_size)]
        for future in tqdm(as_completed(out), file=sys.stdout, total=len(out)):
            for index, tile in future.result():
                hfile.create_dataset(index, data=tile)
    hfile.close()
    
    df['tile_store'] = output_hdf_file
    
    logger.info(df)
    df.to_csv(output_header_file)

    properties = {
        "slide_tiles": output_header_file, # "Tiles" are the metadata that describe them
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
