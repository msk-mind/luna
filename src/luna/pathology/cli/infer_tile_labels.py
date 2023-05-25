# General imports
import logging
import os
import sys
from pathlib import Path

import fire
import fsspec
import pandas as pd
import torch
from fsspec import open
from torch.utils.data import DataLoader
from tqdm import tqdm

from luna.common.custom_logger import init_logger
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.analysis.ml import (
    HDF5Dataset,
    TorchTransformModel,
    post_transform_to_2d,
)

init_logger()
logger = logging.getLogger("infer_tile_labels")


@timed
@save_metadata
def cli(
    tiles_urlpath: str = "???",
    torch_model_repo_or_dir: str = "???",
    model_name: str = "???",
    num_cores: int = 4,
    batch_size: int = 8,
    output_urlpath: str = ".",
    kwargs: dict = {},
    storage_options: dict = {},
    output_storage_options: dict = {},
):
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
    config = get_config(vars())

    df_output = infer_tile_labels(
        config["tiles_urlpath"],
        config["torch_model_repo_or_dir"],
        config["model_name"],
        config["num_cores"],
        config["batch_size"],
        config["kwargs"],
        config["storage_options"],
    )

    fs, output_path_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_file = (
        Path(output_path_prefix) / "tile_scores_and_labels_pytorch_inference.parquet"
    )
    #
    with fs.open(output_file, "wb") as of:
        df_output.to_parquet(of)

    # Save our properties and params
    properties = {
        "slide_tiles": output_file,
        "feature_data": output_file,
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns),
    }

    return properties


def infer_tile_labels(
    tiles_urlpath: str,
    torch_model_repo_or_dir: str,
    model_name: str,
    num_cores: int,
    batch_size: int,
    kwargs: dict,
    storage_options: dict,
):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        input_slide_tiles (str): path to a slide-tile manifest file (.tiles.csv)
        output_urlpath (str): output/working directory
        hub_repo_or_dir (str): repository root name like (namespace/repo) at github.com to serve torch.hub models.
            Or path to a local model
        model_name (str): torch hub model name (a nn.Module at the repo repo_name)
        num_cores (int): Number of cores to use for CPU parallelization
        batch_size (int): size in batch dimension to chuck inference (8-256 recommended, depending on memory usage)
        kwargs (dict): additional keywords to pass to model initialization

    Returns:
        pd.DataFrame: augmented tiles dataframe
    """
    # Get our model and transforms and construct the Tile Dataset and Classifier
    if os.path.exists(torch_model_repo_or_dir):
        source = "local"
    else:
        source = "github"

    logger.info(f"Torch hub source = {source} @ {torch_model_repo_or_dir}")

    if source == "github":
        logger.info(f"Available models: {torch.hub.list(torch_model_repo_or_dir)}")

    ttm = torch.hub.load(
        torch_model_repo_or_dir, model_name, source=source, **kwargs, force_reload=True
    )

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

    with open(tiles_urlpath, **storage_options) as of:
        df = pd.read_parquet(of).reset_index().set_index("address")
    ds = HDF5Dataset(df, preprocess=preprocess)
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
    df_output.index.name = "address"

    logger.info(df_output)
    return df_output


if __name__ == "__main__":
    fire.Fire(cli)
