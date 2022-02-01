# General imports
import os, json, logging, yaml, sys
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('run_stardist_cell_detection')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('cell_expansion_size', float), ('image_type', str), ('output_dir', str), ('debug_opts', str), ('num_cores', int)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-cs', '--cell_expansion_size', required=False,
              help="Size in pixels to expand cell cytoplasm")  
@click.option('-it', '--image_type', required=False,
              help="qupath ImageType")  
@click.option('-db', '--debug_opts', required=False,
              help="qupath ImageType", default='')  
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Run stardist using qupath CLI within a docker container
    
    Note: containers are spawned with a root process, and will not exit if the CLI tool is aborted.
    
    TODO: Improve handling of containers, move to singularity, and/or ensure graceful exits

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
    \b
    Outputs:
        cell_objects
    \b
    Example:
        run_stardist_cell_detection 10001.svs
            -cs 8 -it -it BRIGHTFIELD_H_DAB -nc 8
            -o 10001/cells
    """
    cli_runner( cli_kwargs, _params_, run_stardist_cell_detection)

import docker
from pathlib import Path
import pandas as pd
def run_stardist_cell_detection(input_slide_image, cell_expansion_size, image_type, output_dir, debug_opts, num_cores):
    """Run stardist using qupath CLI

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        cell_expansion_size (float): size in pixels to expand cell cytoplasm
        num_cores (int): Number of cores to use for CPU parallelization
        image_type (str): qupath image type (BRIGHTFIELD_H_DAB)
        output_dir (str): output/working directory
        debug_opts (str): debug options passed as arguments to groovy script

    Returns:
        dict: metadata about function call
    """
    slide_filename = Path(input_slide_image).name
    slide_name = Path(input_slide_image).stem

    logger.info("Launching docker container:")
    logger.info(f"\tvolumes={input_slide_image}:'/inputs/{slide_filename}', {output_dir}:'/output_dir'")
    logger.info(f"\tnano_cpus={int(num_cores * 1e9)}")
    logger.info(f"\timage='mskminddev/qupath-stardist-cuda:latest'")
    logger.info(f"\tcommand=QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy")

    client = docker.from_env()
    container = client.containers.run(
        volumes={input_slide_image: {'bind': f'/inputs/{slide_filename}', 'mode': 'ro'}, os.path.join(output_dir, 'workdir'): {'bind': '/output_dir', 'mode': 'rw'}},
        nano_cpus=int(num_cores * 1e9),
        image='mskminddev/qupath-stardist-cuda:latest', 
        command=f"QuPath script --image /inputs/{slide_filename} --args [cellSize={cell_expansion_size},imageType={image_type},{debug_opts}] /scripts/stardist_simple.groovy",  
        detach=True)

    for line in container.logs(stream=True):
        print (line.decode(), end='')

    stardist_output = os.path.join (output_dir, 'workdir', 'cell_detections.tsv')
    
    df = pd.read_csv(stardist_output, sep='\t')
    df.index = "cell-" + df.index.astype(int).astype(str)
    df.index.rename("cell_id", inplace=True)

    df = df.rename(columns={'Centroid X µm':'x_coord', 'Centroid Y µm': 'y_coord'}) # x,ys follow this convention

    output_header_file = os.path.join (output_dir, f'{slide_name}_cell_objects.csv')
    df.to_csv(output_header_file)

    logger.info("Generated cell data:")
    logger.info(df)

    properties = {
        "cell_objects": output_header_file,
        "spatial": True,
        "total_cells": len(df),
    }

    return properties


if __name__ == "__main__":
    cli()
