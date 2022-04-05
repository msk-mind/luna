# General imports
import os, json, logging, yaml
import click
import requests

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('post_to_dataset') ### Add CLI tool name

from luna.common.utils import cli_runner

_params_ = [('input_feature_data', str), ('waystation_url', str), ('dataset_id', str)]

@click.command()
@click.argument('input_feature_data', nargs=1)
### Additional options
@click.option('-w', '--waystation_url', required=False,
              help='URL of waystation')
@click.option('-dsid', '--dataset_id', required=False,
              help='Dataset identifier (table name)')
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
        post_to_dataset /path/to/featuredata
            --waystation_url Tumor
            --dataset_id MY_DATASET
    """
    cli_runner( cli_kwargs, _params_, post_to_dataset, pass_keys=True)


### Transform imports 
def post_to_dataset(input_feature_data, waystation_url, dataset_id, keys):
    """ CLI tool method

    Args:
        input_data (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """

    segment_id = "-".join(
        [v for _, v in sorted(keys.items())]
    )

    logger.info (f"Input feature data: {input_feature_data}, {keys}")
    
    post_url = os.path.join ( waystation_url, "datasets", dataset_id, "segments", segment_id )

    logger.info (f"Posting to: {post_url}")

    res = requests.post(post_url, files={'segment_data': open (input_feature_data, 'rb')}, data={"segment_keys": json.dumps(keys)})

    print (res.text)

    logger.info (f"Response: {res}, Response data: {res.json()}")
    
    return {}

if __name__ == "__main__":
    cli()
