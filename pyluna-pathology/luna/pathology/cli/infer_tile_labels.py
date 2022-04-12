# General imports
import os
import logging
import sys
import click
import torch
from torch.utils.data import DataLoader
from luna.pathology.analysis.ml import (
    HD5FDataset,
    TorchTransformModel,
    post_transform_to_2d,
)

import pandas as pd
from tqdm import tqdm
from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

init_logger()
logger = logging.getLogger("infer_tile_labels")


_params_ = [
    ("input_slide_tiles", str),
    ("output_dir", str),
    ("hub_repo_or_dir", str),
    ("model_name", str),
    ("kwargs", dict),
    ("num_cores", int),
    ("batch_size", int),
]


@click.command()
@click.argument("input_slide_tiles", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-rn",
    "--hub_repo_or_dir",
    required=False,
    help="repository name to pull model and weight from, e.g. msk-mind/luna-ml",
)
@click.option(
    "-mn", 
    "--model_name", 
    required=False, 
    help="torch hub model name",
)
@click.option(
    "-kw",
    "--kwargs",
    required=False,
    help="additional keywords to pass to model initialization",
    default={},
)
@click.option(
    "-nc", "--num_cores", required=False, help="Number of cores to use", default=4
)
@click.option(
    "-bx",
    "--batch_size",
    required=False,
    help="batch size used for inference speedup",
    default=64,
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
@click.option(
    "-dsid",
    "--dataset_id",
    required=False,
    help="Optional dataset identifier to add results to",
)
def cli(**cli_kwargs):
    """Run a model with a specific pre-transform for all tiles in a slide (tile_images), requires tiles to be saved (save_tiles) first

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
    cli_runner(cli_kwargs, _params_, infer_tile_labels)


def infer_tile_labels(
    input_slide_tiles,
    output_dir,
    hub_repo_or_dir,
    model_name,
    num_cores,
    batch_size,
    kwargs,
):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        output_dir (str): output/working directory
        hub_repo_or_dir (str): repository root name like (namespace/repo) at github.com to serve torch.hub models.
            Or path to a local model
        model_name (str): torch hub model name (a nn.Module at the repo repo_name)
        num_cores (int): Number of cores to use for CPU parallelization
        batch_size (int): size in batch dimension to chuck inference (8-256 recommended, depending on memory usage)
        kwargs (str): additional keywords to pass to model initialization

    Returns:
        dict: metadata about function call
    """
    # Get our model and transforms and construct the Tile Dataset and Classifier

    if os.path.exists(hub_repo_or_dir):
        source = "local"
    else:
        source = "github"

    logger.info(f"Torch hub source = {source} @ {hub_repo_or_dir}")

    if source == "github":
        logger.info(f"Available models: {torch.hub.list(hub_repo_or_dir)}")

    ttm = torch.hub.load(hub_repo_or_dir, model_name, source=source, **kwargs, force_reload=True)

    if not isinstance(ttm, TorchTransformModel):
        raise RuntimeError(f"Not a valid model, loaded model was of type {type(ttm)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device = {device}")

    if isinstance(
        ttm, TorchTransformModel
    ):  # This class packages preprocesing, the model, and optionally class_labels all together
        preprocess = ttm.get_preprocess()
        transform = ttm.transform
        ttm.model.to(device)

    df = pd.read_parquet(input_slide_tiles).reset_index().set_index("address")
    ds = HD5FDataset(df, preprocess=preprocess)
    loader = DataLoader(
        ds, num_workers=num_cores, batch_size=batch_size, pin_memory=True
    )

    # Generate aggregate dataframe
    with torch.no_grad():
        df_scores = pd.concat(
            [
                pd.DataFrame(
                    post_transform_to_2d(transform(data.to(device))), index=index
                )
                for data, index in tqdm(loader, file=sys.stdout)
            ]
        )

    if hasattr(ttm, "column_labels"):
        logger.info(f"Mapping column labels -> {ttm.column_labels}")
        df_scores = df_scores.rename(columns=ttm.column_labels)

    df_output = df.join(df_scores)
    df_output.columns = df_output.columns.astype(str)

    logger.info(df_output)

    output_file = os.path.join(
        output_dir, "tile_scores_and_labels_pytorch_inference.parquet"
    )
    df_output.to_parquet(output_file)

    # Save our properties and params
    properties = {
        "slide_tiles": output_file,
        "feature_data": output_file,
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns),
    }

    return properties


if __name__ == "__main__":
    cli()
