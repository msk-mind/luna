import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('extract_voxels')

from luna.common.utils import cli_runner

_params_ = [('input_itk_volume', str), ('output_dir', str)]

@click.command()
@click.argument('input_itk_volume', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Turns an ITK volume into a numpy volume

    \b
    Inputs:
        input_itk_volume: itk compatible image volume (.mhd, .nrrd, .nii, etc.)
    \b
    Outputs:
        npy_volume
    \b
    Example:
        extract_voxels ./scans/original/NRRDs/10001.nrrd
            -o scans/windowed/NRRDs/10001
    """
    cli_runner(cli_kwargs, _params_, extract_voxels)

import medpy.io
import numpy as np
from pathlib import Path
def extract_voxels(input_itk_volume, output_dir):
    """Save a numpy file from a given ITK volume

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    file_stem = Path(input_itk_volume).stem
    file_ext  = Path(input_itk_volume).suffix

    outFileName = os.path.join(output_dir, file_stem + '.npy')

    image, header = medpy.io.load(input_itk_volume)

    np.save(outFileName, image)

    logger.info (f"Extracted voxels of shape {image.shape}")

    # Prepare metadata and commit
    properties = {
        'npy_volume' : outFileName,
    }

    return properties


if __name__ == "__main__":
    cli()

