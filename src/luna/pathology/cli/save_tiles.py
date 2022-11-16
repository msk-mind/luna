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
@click.option('-dsid', '--dataset_id', required=False,
              help='Optional dataset identifier to add results to')
def cli(**cli_kwargs):
    """Saves tiles to disk
    
    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Adds tile_store to manifest for use in HDF5 Image loader

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles: path to tile images (.tiles.parquet)

    Outputs:
        slide_tiles
    \b
    Example:
        save_tiles 10001.svs 10001/tiles
            -nc 8 -bx 200
            -o 10001/tile_data
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
    Returns address, tile for a list of tile rows (batched)

    Args:
        iterrows (list[pd.Series]): list of rows with tile metadata
        input_slide_image: path to openslide compatible image

    """
    slide = openslide.OpenSlide(str(input_slide_image))
    return [(index, get_tile_from_slide(row, slide)) for index, row in iterrows]

def save_tiles(input_slide_image, input_slide_tiles, output_dir, num_cores, batch_size):
    """Saves tiles to disk
    
    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.parquet)
        output_dir (str): output/working directory
        batch_size (int): size in batch dimension to chuck jobs

    Returns:
        dict: metadata about function call
    """
    slide_id = Path(input_slide_image).stem
    df = pd.read_parquet(input_slide_tiles).reset_index().set_index('address')

    output_header_file = f"{output_dir}/{slide_id}.tiles.parquet"
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
    df.to_parquet(output_header_file)

    properties = {
        "slide_tiles": output_header_file, # "Tiles" are the metadata that describe them
        "feature_data": output_header_file, # Tiles can act like feature data
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
