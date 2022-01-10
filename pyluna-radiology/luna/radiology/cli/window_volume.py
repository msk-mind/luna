# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('window_volume')

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str), ('low_level', float), ('high_level', float)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-ll', '--low_level', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-hl', '--high_level', required=False,
              help="torch hub transform name")   
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Run with explicit arguments:

    \b
        infer_tiles
            -i 1412934/data/TileImages
            -o 1412934/data/TilePredictions
            -rn msk-mind/luna-ml:main 
            -tn tissue_tile_net_transform 
            -mn tissue_tile_net_model_5_class
            -wt main:tissue_net_2021-01-19_21.05.24-e17.pth

    Run with implicit arguments:

    \b
        infer_tiles -m 1412934/data/TilePredictions/metadata.json
    
    Run with mixed arguments (CLI args override yaml/json arguments):

    \b
        infer_tiles --input_data 1412934/data/TileImages -m 1412934/data/TilePredictions/metadata.json
    """
    cli_runner(cli_kwargs, _params_, window_volume)

import medpy.io
import numpy as np
from pathlib import Path
def window_volume(input_data, output_dir, low_level, high_level):

        os.makedirs(output_dir, exist_ok=True)
        
        file_stem = Path(input_data).stem
        file_ext  = Path(input_data).suffix

        outFileName = os.path.join(output_dir, file_stem + '.windowed' + file_ext)

        logger.info ("Applying window [%s,%s]", low_level, high_level)

        image, header = medpy.io.load(input_data)
        image = np.clip(image, low_level, high_level )
        medpy.io.save(image, outFileName, header)
        # Prepare metadata and commit
        properties = {
            'data' : outFileName,
        }

        return properties


if __name__ == "__main__":
    cli()

