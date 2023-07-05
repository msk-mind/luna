# General imports
import os
import ssl
import sys
from pathlib import Path
from typing import Optional

import fire
import fsspec
import pandas as pd
import torch
from dask.distributed import Client
from fsspec import open
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.analysis.ml import (
    HDF5Dataset,
    TorchTransformModel,
    post_transform_to_2d,
)
from luna.pathology.cli.save_tiles import save_tiles


@timed
@save_metadata
def cli(
    slide_urlpath: str = "",
    tiles_urlpath: str = "",
    tile_size: Optional[int] = None,
    requested_magnification: Optional[int] = None,
    torch_model_repo_or_dir: str = "???",
    model_name: str = "???",
    num_cores: int = 4,
    batch_size: int = 8,
    output_urlpath: str = ".",
    kwargs: dict = {},
    use_gpu: bool = False,
    dask_options: dict = {},
    insecure: bool = False,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with TiffSlide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): path to a slide-tile manifest file (.tiles.csv)
        tile_size (int): size of tiles to use (at the requested magnification)
        torch_model_repo_or_dir (str): repository root name like (namespace/repo) at github.com to serve torch.hub models. Or path to a local model (e.g. msk-mind/luna-ml)
        model_name (str): torch hub model name (a nn.Module at the repo repo_name)
        num_cores (int): Number of cores to use for CPU parallelization
        batch_size (int): size in batch dimension to chuck inference (8-256 recommended, depending on memory usage)
        output_urlpath (str): output/working directory
        kwargs (dict): additional keywords to pass to model initialization
        use_gpu (bool): use GPU if available
        dask_options (dict): options to pass to dask client
        insecure (bool): insecure SSL
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        dict: metadata
    """
    config = get_config(vars())

    if not config["slide_urlpath"] and not config["tiles_urlpath"]:
        raise fire.core.FireError("Specify either tiles_urlpath or slide_urlpath")

    if not config["tile_size"] and not config["tiles_urlpath"]:
        raise fire.core.FireError("Specify either tiles_urlpath or tile_size")

    df_output = infer_tile_labels(
        config["slide_urlpath"],
        config["tiles_urlpath"],
        config["tile_size"],
        config["requested_magnification"],
        config["torch_model_repo_or_dir"],
        config["model_name"],
        config["num_cores"],
        config["batch_size"],
        config["output_urlpath"],
        config["kwargs"],
        config["use_gpu"],
        config["dask_options"],
        config["insecure"],
        config["storage_options"],
        config["output_storage_options"],
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
    slide_urlpath: str,
    tiles_urlpath: str,
    tile_size: Optional[int],
    requested_magnification: Optional[int],
    torch_model_repo_or_dir: str,
    model_name: str,
    num_cores: int,
    batch_size: int,
    output_urlpath: str,
    kwargs: dict,
    use_gpu: bool,
    dask_options: dict,
    insecure: bool,
    storage_options: dict,
    output_storage_options: dict,
):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with TiffSlide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): path to a slide-tile manifest file (.tiles.parquet)
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation
        torch_model_repo_or_dir (str): repository root name like (namespace/repo) at github.com to serve torch.hub models. Or path to a local model (e.g. msk-mind/luna-ml)
        model_name (str): torch hub model name (a nn.Module at the repo repo_name)
        num_cores (int): Number of cores to use for CPU parallelization
        batch_size (int): size in batch dimension to chuck inference (8-256 recommended, depending on memory usage)
        output_urlpath (str): output/working directory
        kwargs (dict): additional keywords to pass to model initialization
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        pd.DataFrame: augmented tiles dataframe
    """

    if insecure:
        ssl._create_default_https_context = ssl._create_unverified_context

    # Get our model and transforms and construct the Tile Dataset and Classifier
    if os.path.exists(torch_model_repo_or_dir):
        source = "local"
    else:
        source = "github"

    logger.info(f"Torch hub source = {source} @ {torch_model_repo_or_dir}")

    # if source == "github":
    # logger.info(f"Available models: {torch.hub.list(torch_model_repo_or_dir, trust_repo=False)}")

    ttm = torch.hub.load(
        torch_model_repo_or_dir,
        model_name,
        source=source,
        **kwargs,
        force_reload=True,
        trust_repo=True,
    )

    if not isinstance(ttm, TorchTransformModel):
        raise RuntimeError(f"Not a valid model, loaded model was of type {type(ttm)}")

    # load/generate tiles
    if tiles_urlpath:
        with open(tiles_urlpath, **storage_options) as of:
            df = pd.read_parquet(of).reset_index().set_index("address")
    elif tile_size is not None:
        Client(**dask_options)
        slide_id = Path(slide_urlpath).stem
        tiles_urlpath = str(Path(output_urlpath) / f"{slide_id}.tiles.h5")

        df = save_tiles(
            slide_urlpath,
            tile_size,
            tiles_urlpath,
            batch_size,
            requested_magnification,
            storage_options,
            output_storage_options,
        )
    else:
        raise RuntimeError(
            "Need to specify tiles_urlpath or both slide_urlpath and tile_size"
        )

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    logger.info(f"Using device = {device}")

    preprocess = ttm.get_preprocess()
    transform = ttm.transform
    ttm.model.to(device)

    ds = HDF5Dataset(df, preprocess=preprocess, storage_options=storage_options)
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


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
