# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('infer_tille_labels')

from luna.common.utils import cli_runner

import torch
from torch.utils.data import DataLoader
from luna.pathology.common.ml import BaseTorchTileDataset, BaseTorchTileClassifier

import pandas as pd
from tqdm import tqdm

_params_ = [('input_slide_tiles', str), ('output_dir', str), ('repo_name', str), ('transform_name', str), ('model_name', str), ('weight_tag', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.argument('input_slide_tiles', nargs=1)
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
              help="batch size used for inference speedup", default=64)    
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Run a model with a specific pre-transform for all tiles in a slide (tile_images)

    \b
    Inputs:
        input_slide_tiles: path to tile images (.tiles.pil)
    \b
    Outputs:
        tile_scores
    \b
    Example:
        infer_tiles tiles/slide-100012/tiles
            -rn msk-mind/luna-ml:main 
            -mn tissue_tile_net_model_5_class
            -tn tissue_tile_net_transform 
            -wt main:tissue_net_2021-01-19_21.05.24-e17.pth
            -o tiles/slide-100012/scores
    """
    cli_runner( cli_kwargs, _params_, infer_tile_labels)

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

def infer_tile_labels(input_slide_tiles, output_dir, repo_name, transform_name, model_name, weight_tag, num_cores, batch_size):
    """Generate tile addresses, scores and optionally annotation labels using models stored in torch.hub format

    Args:
        input_slide_tiles (str): path to application configuration file.
        output_dir (str): datastore name. usually a slide id.
        repo_name (str): method parameters including input, output details.
        transform_name (str):
        model_name (str):
        weight_tag (str):

    Returns:
        None
    """
    # Get our model and transforms and construct the Tile Dataset and Classifier
    logger.info(f"Loading model and transform: repo_name={repo_name}, transform_name={transform_name}, model_name={model_name}")
    logger.info(f"Using weights weight_tag={weight_tag}")

    tile_dataset     = TileDatasetGithub(tile_path=input_slide_tiles, repo_name=repo_name, transform_name=transform_name)
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
    properties = {
        "tile_scores": output_file,
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns),
    }

    return properties


if __name__ == "__main__":
    cli()
