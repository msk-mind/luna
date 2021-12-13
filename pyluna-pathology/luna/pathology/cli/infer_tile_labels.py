# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('infer_tille_labels')

from luna.common.utils import validate_params

import torch
from torch.utils.data import DataLoader
from luna.pathology.common.ml import BaseTorchTileDataset, BaseTorchTileClassifier

import pandas as pd
from tqdm import tqdm

_params = [('input_data', str), ('output_dir', str), ('repo_name', str), ('transform_name', str), ('model_name', str), ('weight_tag', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rn', '--repo_name', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-tn', '--transform_name', required=False,
              help="torch hub transform name")   
@click.option('-mn', '--model_name', required=False,
              help="torch hub model name")    
@click.option('-wt', '--weight_tag', required=False,
              help="weight tag filename")  
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-bx', '--batch_size', required=False,
              help="weight tag filename", default=256)    
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Run with explicit arguments:

    \b
        infer_tile_labels
            -i 1412934/data/TileImages
            -o 1412934/data/TilePredictions
            -r msk-mind/luna-ml:main 
            -t tissue_tile_net_transform 
            -m tissue_tile_net_model_5_class
            -w main:tissue_net_2021-01-19_21.05.24-e17.pth

    Run with implicit arguments:

    \b
        infer_tile_labels -m 1412934/data/TilePredictions/metadata.json
    
    Run with mixed arguments (CLI args override yaml/json arguments):
    
    \b
        infer_tile_labels --input_data 1412934/data/TileImages -m 1412934/data/TilePredictions/metadata.json
    """
    kwargs = {}

    # Get params from param file
    if cli_kwargs.get('method_param_path'):
        with open(cli_kwargs.get('method_param_path'), 'r') as yaml_file:
            yaml_kwargs = yaml.safe_load(yaml_file)
        kwargs.update(yaml_kwargs) # Fill from json
    
    for key in list(cli_kwargs.keys()):
        if cli_kwargs[key] is None: del cli_kwargs[key]

    # Override with CLI arguments
    kwargs.update(cli_kwargs) # 

    # Validate them
    kwargs = validate_params(kwargs, _params)

    infer_tile_labels(**kwargs)

# We are acting a bit like a consumer of the base classes here-
class TileDatasetGithub(BaseTorchTileDataset):
    def setup(self, repo_name, transform_name):
        self.transform = torch.hub.load(repo_name, transform_name)
    def preprocess(self, input_tile):
        return self.transform(input_tile)
    
class TileClassifierGithub(BaseTorchTileClassifier):
    def setup(self, repo_name, model_name, weight_tag):
        self.model = torch.hub.load(repo_name, model_name, weight_tag=weight_tag)
    def predict(self, input_tiles):
        return self.model(input_tiles)

def infer_tile_labels(input_data, output_dir, repo_name, transform_name, model_name, weight_tag, num_cores, batch_size):
    """Generate tile addresses, scores and optionally annotation labels using models stored in torch.hub format

    Args:
        input_data (str): path to application configuration file.
        output_dir (str): datastore name. usually a slide id.
        repo_name (str): method parameters including input, output details.
        transform_name (str):
        model_name (str):
        weight_tag (str):

    Returns:
        None
    """
    input_params = validate_params(locals(), _params) # Capture input parameters as dict
    os.makedirs(output_dir, exist_ok=True)

    # Get our model and transforms and construct the Tile Dataset and Classifier
    logger.info(f"Loading model and transform: repo_name={repo_name}, transform_name={transform_name}, model_name={model_name}")
    logger.info(f"Using weights weight_tag={weight_tag}")

    tile_dataset     = TileDatasetGithub(tile_path=input_data, repo_name=repo_name, transform_name=transform_name)
    tile_classifier  = TileClassifierGithub(repo_name=repo_name, model_name=model_name, weight_tag=weight_tag)

    tile_loader = DataLoader(tile_dataset, num_workers=num_cores, batch_size=batch_size, pin_memory=True)

    # Generate aggregate dataframe
    with torch.no_grad():
        df_scores = pd.concat([tile_classifier(index, data) for index, data in tqdm(tile_loader)])
        
    if hasattr(tile_classifier.model, 'class_labels'):
        logger.info(f"Mapping column labels -> {tile_classifier.model.class_labels}")
        df_scores = df_scores.rename(columns=tile_classifier.model.class_labels)

    df_output = tile_dataset.tile_manifest.join(df_scores)    

    logger.info(df_output)

    output_file = os.path.join(output_dir, "tile_scores_and_labels_pytorch_inference.csv")
    df_output.to_csv(output_file)

    # Save our properties and params
    extra_props = {
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns)
    }

    input_params.update(extra_props)

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(input_params, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    cli()
