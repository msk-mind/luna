import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('window_volume')

from luna.common.utils import cli_runner

_params_ = [('input_itk_volume', str), ('output_dir', str), ('low_level', float), ('high_level', float)]

@click.command()
@click.argument('input_itk_volume', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-ll', '--low_level', required=False,
              help="lower bound of window")
@click.option('-hl', '--high_level', required=False,
              help="upper bound of window")   
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """
    Applies a window function to an input itk volume, outputs windowed volume

    \b
    Inputs:
        input_itk_volume: itk compatible image volume (.mhd, .nrrd, .nii, etc.)
    \b
    Outputs:
        itk_volume
    \b
    Example:
        window_volume ./scans/original/NRRDs/10001.nrrd
            --low_level 0
            --high_level 250
            -o scans/windowed/NRRDs/10001
    """
    cli_runner(cli_kwargs, _params_, window_volume)

import medpy.io
import numpy as np
from pathlib import Path
def window_volume(input_itk_volume: str, output_dir: str, low_level: float, high_level: float):
    """Applies a window function (clipping) to an input itk volume, outputs windowed volume 

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        output_dir (str): output/working directory
        low_level (float): lower bound of clipping operation
        high_level (float): higher bound of clipping operation
    
    Returns:
        dict: metadata about function call
    """
    file_stem = Path(input_itk_volume).stem
    file_ext  = Path(input_itk_volume).suffix

    outFileName = os.path.join(output_dir, file_stem + '.windowed' + file_ext)

    logger.info ("Applying window [%s,%s]", low_level, high_level)

    image, header = medpy.io.load(input_itk_volume)
    image = np.clip(image, low_level, high_level )
    medpy.io.save(image, outFileName, header)
    # Prepare metadata and commit
    properties = {
        'itk_volume' : outFileName,
    }

    return properties


if __name__ == "__main__":
    cli()

