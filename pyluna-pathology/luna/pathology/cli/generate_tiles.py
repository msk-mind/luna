# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_tiles')

from luna.common.utils import cli_runner

import pandas as pd
from tqdm import tqdm

_params_ = [('input_slide_image', str), ('output_dir', str), ('requested_tile_size', int), ('batch_size', int), ('requested_magnification', float), ('num_cores', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-rts', '--requested_tile_size', required=False,
              help="Number of cores to use")
@click.option('-rmg', '--requested_magnification', required=False,
              help="Number of cores to use")
@click.option('-bx', '--batch_size', required=False,
              help="batch size used for inference speedup", default=64)    
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
    cli_runner( cli_kwargs, _params_, generate_tiles)

import openslide
from luna.pathology.common.preprocess import get_scale_factor_at_magnfication, get_full_resolution_generator, coord_to_address, address_to_coord
import itertools
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return [[entry for entry in iterable if entry is not None]
            for iterable in itertools.zip_longest(*args, fillvalue=fillvalue)]

def get_tile_bytes(indices, input_slide_image, full_resolution_tile_size, requested_tile_size ):
    slide = openslide.OpenSlide(str(input_slide_image))
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)
    return [(index, full_generator.get_tile(full_level, address_to_coord(index)).resize((requested_tile_size,requested_tile_size)).tobytes()) for index in indices]

def generate_tiles(input_slide_image, requested_tile_size, requested_magnification, output_dir, num_cores, batch_size):
    """Generate tile addresses, scores and optionally annotation labels using models stored in torch.hub format

    Args:
        input_slide_image: slide image (.svs)
        output_dir (str): output/working directory
        num_cores (int): how many cores to use for dataloading

    Returns:
        dict: metadata about function call
    """
    slide_name = Path(input_slide_image).stem

    slide = openslide.OpenSlide(str(input_slide_image))
    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])

    to_mag_scale_factor         = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    if not to_mag_scale_factor % 1 == 0: 
        raise ValueError("You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size = requested_tile_size * to_mag_scale_factor
    logger.info("Normalized magnification scale factor for %sx is %s", requested_magnification, to_mag_scale_factor)
    logger.info("Requested tile size=%s, tile size at full magnficiation=%s", requested_tile_size, full_resolution_tile_size)

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)

    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    logger.info("tiles x %s, tiles y %s", tile_x_count, tile_y_count)

    # populate address, coordinates
    address_raster = [{"address": coord_to_address(address, requested_magnification), "coordinates": address}
                      for address in itertools.product(range(1, tile_x_count-1), range(1, tile_y_count-1))]
    logger.info("Number of tiles in raster: %s", len(address_raster))

    df = pd.DataFrame(address_raster).set_index("address")

    df.loc[:, 'tile_image_offset']  = None

    output_binary_file = f"{output_dir}/{slide_name}.tiles.pil"
    output_header_file = f"{output_dir}/{slide_name}.tiles.csv"

    fp = open(output_binary_file, 'wb')
    offset = 0
    counter = 0
    logger.info(f"Now generating tiles with num_cores={num_cores} and batch_size={batch_size}!")
    with ProcessPoolExecutor(num_cores) as executor:
        out = [executor.submit(get_tile_bytes, index, input_slide_image, full_resolution_tile_size, requested_tile_size ) for index in grouper(df.index, batch_size)]
        for future in tqdm(as_completed(out), file=sys.stdout, total=len(out)):
            # if counter % 10000 == 0: logger.info( "Proccessing tiles [%s,%s]", counter, len(df))
            for index, tile in future.result(): 
                fp.write( tile )
                
                df.loc[index, "tile_image_offset"]   = int(offset)

                offset += len(tile)
                counter+=1
    fp.close()

    df.loc[:, 'tile_image_binary'] = output_binary_file
    df.loc[:, 'tile_image_length']  = 3 * requested_tile_size ** 2
    df.loc[:, 'tile_image_size_xy'] = requested_tile_size
    df.loc[:, 'tile_image_mode'] = 'RGB'

    logger.info (df)        

    df.to_csv(output_header_file)

    properties = {
        "slide_tiles": output_header_file,
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
