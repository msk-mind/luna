# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('extract_kfunction')

from luna.common.utils import cli_runner

_params_ = [('input_cell_objects', str), ('tile_size', int), ('intensity_label', str), ('radius', float), ('tile_stride', int), ('num_cores', int), ('output_dir', str)]

@click.command()
@click.argument('input_cell_objects', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-il', '--intensity_label', required=False,
              help="Intensity values to use")
@click.option('-r', '--radius', required=False,
              help="ik-function radius")
@click.option('-rts', '--tile_size', required=False,
              help="Tile size in μm, window dimensions")
@click.option('-rtd', '--tile_stride', required=False,
              help="Tile stride in μm, how far the next tile/window is from the last")
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Run k function using a sliding window approach, where the k-function is computed locally in a smaller window, and aggregated across the entire slide.
    
    Additionally runs the IK function, which is a special version of the normal K function, senstive to the intensity of staining withink the K-function radius.

    Generates "super tiles" with the windowed statistics for downstream processing.

    \b
    Inputs:
        input_cell_objects: cell objects (.csv)
    \b
    Outputs:
        slide_tiles
    \b
    Example:
        extract_kfunction 10001/cells/objects.csv
            -rts 300 -rtd 300 -nc 32 -r 160 -il 'DAB: Cytoplasm: Mean'
            -o 10001/spatial_features/
    """
    cli_runner( cli_kwargs, _params_, extract_kfunction)

from pathlib import Path
import pandas as pd
from luna.pathology.spatial.stats import Kfunction
import numpy as np

from luna.pathology.common.utils import coord_to_address
from concurrent.futures import ProcessPoolExecutor

from tqdm.contrib.itertools import product

def extract_kfunction(input_cell_objects, tile_size, intensity_label, tile_stride, radius, num_cores, output_dir):
    """Run k function using a sliding window approach, where the k-function is computed locally in a smaller window, and aggregated across the entire slide.

    Args:
        input_cell_objects (str): path to cell objects (.csv)
        output_dir (str): output/working directory
        tile_size (int): size of tiles to use (at the requested magnification)
        tile_stride (int): spacing between tiles
        distance_scale (float): scale at which to consider k-function
        num_cores (int): Number of cores to use for CPU parallelization
        intensity_label (str): Columns of cell object to use for intensity calculations

    Returns:
        dict: metadata about function call
    """
    df = pd.read_parquet(input_cell_objects)

    l_address = []
    l_k_function = []
    l_x_coord = []
    l_y_coord = []

    feature_name = f"ikfunction_r{radius}_stain{intensity_label.replace(' ','_').replace(':','')}"

    coords = product(range(int(df['x_coord'].min()), int(df['x_coord'].max()), tile_stride), range(int(df['y_coord'].min()), int(df['y_coord'].max()), tile_stride))
    
    logger.info("Submitting tasks...")
    with ProcessPoolExecutor(num_cores) as executor:
        for x, y in coords:
            df_tile = df.query(f'x_coord >= {x} and x_coord <= {x+tile_size} and y_coord >={y} and y_coord <= {y+tile_size}')

            if len(df_tile) < 3: continue

            out = executor.submit(Kfunction, 
                df_tile[['x_coord', 'y_coord']], 
                df_tile[['x_coord', 'y_coord']], 
                intensity=np.array(df_tile[intensity_label]),
                radius=radius, count=True)

            l_address.append(coord_to_address((x,y), 0))
            l_k_function.append(out)
            l_x_coord.append(x)
            l_y_coord.append(y)
        logger.info("Waiting for all tasks to complete...")
        
    df_stats = pd.DataFrame({'address': l_address, 'x_coord':l_x_coord, 'y_coord':l_y_coord, 'results': l_k_function}).set_index('address')
    df_stats.loc[:, 'xy_extent']  = tile_size 
    df_stats.loc[:, 'tile_size']  = tile_size # Same, 1 to 1
    df_stats.loc[:, 'tile_units'] = 'um' # Same, 1 to 1

    df_stats[feature_name]            = df_stats['results'].apply(lambda x: x.result()['intensity'])
    df_stats[feature_name + '_norm']  = df_stats[feature_name]  / df_stats[feature_name].max()

    df_stats = df_stats.drop(columns=['results']).dropna()

    logger.info("Generated k-function feature data:")
    logger.info (df_stats)
    
    output_tile_header = os.path.join(output_dir, Path(input_cell_objects).stem + '_kfunction_supertiles.parquet')
    df_stats.to_parquet(output_tile_header)

    properties = {
        'slide_tiles': output_tile_header,
    }

    return properties




if __name__ == "__main__":
    cli()
