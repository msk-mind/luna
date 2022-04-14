# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger() ### Add CLI tool name

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str)]

@click.command()
@click.argument('input_data', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
### Additional options
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ A cli tool

    \b
    Inputs:
        input: input data
    \b
    Outputs:
        output data
    \b
    Example:
        CLI_TOOL ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    cli_runner( cli_kwargs, _params_, transform_method)


### Transform imports 
def transform_method(input_data, output_dir):
    """ CLI tool method

    Args:
        input_data (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """

    properties = {

    }

    return properties

if __name__ == "__main__":
    cli()
