# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('run_stardist_cell_detection')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('output_dir', str), ('num_cores', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Run stardist using qupath CLI

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    Outputs:
        cell_objects
    \b
    Example:
        run_stardist_cell_detection 10001.svs
            -nc 8
            -o 10001/cells
    """
    cli_runner( cli_kwargs, _params_, run_stardist_cell_detection)

import docker
from pathlib import Path
def run_stardist_cell_detection(input_slide_image, output_dir, num_cores):
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        num_cores (int): Number of cores to use for CPU parallelization
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    slide_filename = Path(input_slide_image).name

    client = docker.from_env()
    container = client.containers.run(
        volumes={input_slide_image: {'bind': f'/inputs/{slide_filename}', 'mode': 'ro'}, output_dir: {'bind': '/output_dir', 'mode': 'rw'}},
        nano_cpus=int(num_cores * 1e9),
        image='qupath', 
        command=f"QuPath script --image /inputs/{slide_filename} /scripts/stardist_simple.groovy",  
        detach=True)

    for line in container.logs(stream=True):
        print (line.decode(), end='')

    properties = {
        "slide_tiles": output_header_file,
        "total_tiles": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
