# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_tiles')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('output_dir', str), ('tile_size', int),  ('requested_magnification', float)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rts', '--tile_size', required=False,
              help="Size of tiles")
@click.option('-rmg', '--requested_magnification', required=False,
              help="Magnification at which to generate tiles")
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Rasterize a slide into smaller tiles, saving tile metadata as rows in a csv file

    Necessary data for the manifest file are:
    address, x_coord, y_coord, xy_extent, tile_size, tile_units
    
    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    Outputs:
        slide_tiles
    \b
    Example:
        generate_tiles 10001.svs
            -rts 244 -rmg 10
            -o 10001/tiles
    """
    cli_runner( cli_kwargs, _params_, generate_tiles)

import pandas as pd
import openslide
import itertools
from pathlib import Path

from luna.pathology.common.utils import get_scale_factor_at_magnfication, get_full_resolution_generator, coord_to_address

def generate_tiles(input_slide_image, tile_size, requested_magnification, output_dir):
    """Rasterize a slide into smaller tiles
    
    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.csv) is also generated

    Necessary data for the manifest file are:
    address, tile_image_file, full_resolution_tile_size, tile_image_size_xy

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    slide_id = Path(input_slide_image).stem

    slide = openslide.OpenSlide(str(input_slide_image))
    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])

    to_mag_scale_factor         = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    if not to_mag_scale_factor % 1 == 0: 
        logger.error(f"Bad magnficiation scale factor = {to_mag_scale_factor}")
        raise ValueError("You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size = int (tile_size * to_mag_scale_factor)
    logger.info("Normalized magnification scale factor for %sx is %s", requested_magnification, to_mag_scale_factor)
    logger.info("Requested tile size=%s, tile size at full magnficiation=%s", tile_size, full_resolution_tile_size)

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)

    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    logger.info("tiles x %s, tiles y %s", tile_x_count, tile_y_count)

    # populate address, coordinates
    address_raster = [{
        "address": coord_to_address(address, requested_magnification), 
        "x_coord": (address[0]) * full_resolution_tile_size, 
        "y_coord": (address[1]) * full_resolution_tile_size}
                      for address in itertools.product(range(1, tile_x_count-1), range(1, tile_y_count-1))]
                      
    logger.info("Number of tiles in raster: %s", len(address_raster))

    df = pd.DataFrame(address_raster).set_index("address")
    df['xy_extent']  = full_resolution_tile_size
    df['tile_size']  = tile_size
    df['tile_units'] = 'px' # tile coordiates correspond to pixels at max resolution

    logger.info(df)

    output_header_file = f"{output_dir}/{slide_id}.tiles.parquet"
    df.to_parquet(output_header_file)

    properties = {
        "slide_tiles": output_header_file, # "Tiles" are the metadata that describe them
        "total_tiles": len(df),
        "segment_keys": {"slide_id": str(slide_id)}
    }

    return properties


if __name__ == "__main__":
    cli()
