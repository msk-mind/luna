# General imports
import os, json, logging, yaml, sys
from numpy import isin
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('infer_tile_labels')

from luna.common.utils import cli_runner

_params_ = [('input_slide_tiles', str), ('output_dir', str), ('hub_repo_or_dir', str), ('model_name', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.argument('input_slide_tiles', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rn', '--hub_repo_or_dir', required=False,
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
        input_slide_tiles: path to tile images (.tiles.csv)
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

import torch
from torch import nn
from torch.utils.data import DataLoader
from luna.pathology.common.ml import HD5FDataset, TorchTransformModel

import pandas as pd
from tqdm import tqdm
def infer_tile_labels(input_slide_tiles, output_dir, hub_repo_or_dir, model_name, num_cores, batch_size):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        output_dir (str): output/working directory
        repo_name (str): repository root name like (namespace/repo) at github.com to serve torch.hub models
        transform_name (str): torch hub transform name (a function at the repo repo_name)
        model_name (str): torch hub model name (a nn.Module at the repo repo_name)
        weight_tag (str): what weight tag to use
        num_cores (int): Number of cores to use for CPU parallelization 
        batch_size (int): size in batch dimension to chuck inference (8-256 recommended, depending on memory usage)

    Returns:
        dict: metadata about function call
    """
    # Get our model and transforms and construct the Tile Dataset and Classifier

    if os.path.exists(hub_repo_or_dir):
        source = 'local'
    else:
        source = 'github'

    clf = torch.hub.load(hub_repo_or_dir, model_name, source=source)

    if not (isinstance(clf, nn.Module) or isinstance(clf, TorchTransformModel)):
        raise RuntimeError("Not a valid model!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device={device}")

    if isinstance(clf, TorchTransformModel):
        preprocess = clf.get_preprocess()
        transform  = clf.transform
        clf.model.to(device)
    else:
        preprocess = nn.Identity()
        transform = clf.to(device)

    df     = pd.read_csv(input_slide_tiles).set_index('address')
    ds     = HD5FDataset(df, preprocess=preprocess)
    loader = DataLoader(ds, num_workers=num_cores, batch_size=batch_size, pin_memory=True)

    # Generate aggregate dataframe
    with torch.no_grad():
        df_scores = pd.concat([pd.DataFrame(transform(data.to(device)), index=index) for data, index in tqdm(loader, file=sys.stdout)])
        
    if hasattr(clf, 'class_labels'):
        logger.info(f"Mapping column labels -> {clf.class_labels}")
        df_scores = df_scores.rename(columns=clf.class_labels)

    df_output = df.join(df_scores)    

    logger.info(df_output)

    output_file = os.path.join(output_dir, "tile_scores_and_labels_pytorch_inference.csv")
    df_output.to_csv(output_file)

    # Save our properties and params
    properties = {
        "slide_tiles": output_file,
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns),
    }

    return properties


if __name__ == "__main__":
    cli()
