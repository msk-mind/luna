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
from dask.distributed import progress
from loguru import logger
from pandera.typing import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from luna.common.dask import configure_dask_client, get_or_create_dask_client
from luna.common.utils import get_config, make_temp_directory, save_metadata, timed
from luna.pathology.analysis.ml import (
    HDF5Dataset,
    TorchTransformModel,
    post_transform_to_2d,
)
from luna.pathology.cli.generate_tiles import __generate_tiles
from luna.pathology.cli.run_tissue_detection import __detect_tissue, detect_tissue
from luna.pathology.cli.save_tiles import _save_tiles, save_tiles


@timed
@save_metadata
def cli(
    slide_urlpath: str = "",
    tiles_urlpath: str = "",
    tile_size: Optional[int] = None,
    filter_query: str = "",
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
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
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

    if config["slide_urlpath"]:
        slide_id = Path(config["slide_urlpath"]).stem
    else:
        slide_id = Path(config["tiles_urlpath"]).stem.removesuffix(".tiles")

    tiles_urlpath = config["tiles_urlpath"]
    with make_temp_directory() as temp_dir:
        if not tiles_urlpath:
            tiles_result = __generate_tiles(
                config["slide_urlpath"],
                config["tile_size"],
                (Path(temp_dir) / "generate_tiles").as_uri(),
                config["tile_magnification"],
                config["storage_options"],
            )
            detect_tissue_result = __detect_tissue(
                config["slide_urlpath"],
                tiles_result["tiles_url"],
                slide_id,
                config["thumbnail_magnification"],
                config["filter_query"],
                config["batch_size"],
                (Path(temp_dir) / "detect_tissue").as_uri(),
                config["storage_options"],
            )
            save_tiles_result = _save_tiles(
                detect_tissue_result["tiles_urlpath"],
                config["slide_urlpath"],
                (Path(temp_dir) / "save_tiles").as_uri(),
                config["batch_size"],
                config["storage_options"],
            )
            tiles_urlpath = save_tiles_result["tiles_url"]

        return __infer_tile_labels(
            tiles_urlpath,
            slide_id,
            config["output_urlpath"],
            config["torch_model_repo_or_dir"],
            config["model_name"],
            config["num_cores"],
            config["batch_size"],
            config["kwargs"],
            config["use_gpu"],
            config["insecure"],
            config["storage_options"],
            config["output_storage_options"],
        )


def infer_tile_labels(
    slide_manifest: DataFrame,
    tile_size: Optional[int] = None,
    filter_query: str = "",
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    torch_model_repo_or_dir: str = "",
    model_name: str = "",
    num_cores: int = 1,
    batch_size: int = 2000,
    output_urlpath: str = ".",
    kwargs: dict = {},
    use_gpu: bool = False,
    dask_options: dict = {},
    insecure: bool = False,
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> pd.DataFrame:
    client = get_or_create_dask_client()
    configure_dask_client(**dask_options)

    if "tiles_url" not in slide_manifest.columns:
        if tile_size is None:
            raise RuntimeError("Need to have generated tiles or specify tile_size")
        # generate tiles
        slide_manifest = detect_tissue(
            slide_manifest,
            None,
            tile_size=tile_size,
            thumbnail_magnification=thumbnail_magnification,
            tile_magnification=tile_magnification,
            filter_query=filter_query,
            batch_size=batch_size,
            storage_options=storage_options,
            output_urlpath=output_urlpath,
            output_storage_options=output_storage_options,
        )

        slide_manifest = save_tiles(
            slide_manifest,
            output_urlpath,
            batch_size,
            storage_options,
            output_storage_options,
        )

    futures = []
    for row in slide_manifest.itertuples(name="Slide"):
        future = client.submit(
            __infer_tile_labels,
            row.tiles_url,
            row.id,
            output_urlpath,
            torch_model_repo_or_dir,
            model_name,
            num_cores,
            batch_size,
            kwargs,
            use_gpu,
            insecure,
            storage_options,
            output_storage_options,
        )
        futures.append(future)

    progress(futures)
    results = client.gather(futures)
    for idx, result in results.enumerate():
        slide_manifest.at[idx, "tiles_url"] = result
    return slide_manifest


def __infer_tile_labels(
    tiles_urlpath: str,
    slide_id: str,
    output_urlpath: str,
    torch_model_repo_or_dir: str,
    model_name: str,
    num_cores: int,
    batch_size: int,
    kwargs: dict,
    use_gpu: bool,
    insecure: bool,
    storage_options: dict,
    output_storage_options: dict,
):
    """Run inference using a model and transform definition (either local or using torch.hub)

    Decorates existing slide_tiles with additional columns corresponding to class prediction/scores from the model

    Args:
        tiles_urlpath (str): path to a slide-tile manifest file (.tiles.parquet)
        tile_size (int): size of tiles to use (at the requested magnification)
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        requested_magnification (Optional[int]): Magnification scale at which to perform computation
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

    ofs, output_path_prefix = fsspec.core.url_to_fs(
        output_urlpath,
        **output_storage_options,
    )

    output_file = str(Path(output_path_prefix) / f"{slide_id}.tiles.parquet")

    if ofs.exists(output_file):
        logger.info(f"outputs already exist: {output_file}")
        return

    tiles_df = (
        pd.read_parquet(tiles_urlpath, storage_options=storage_options)
        .reset_index()
        .set_index("address")
    )

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

    pin_memory = False
    if use_gpu and torch.cuda.is_available():
        pin_memory = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device = {device}")

    preprocess = ttm.get_preprocess()
    transform = ttm.transform
    ttm.model.to(device)

    ds = HDF5Dataset(tiles_df, preprocess=preprocess, storage_options=storage_options)
    loader = DataLoader(
        ds, num_workers=num_cores, batch_size=batch_size, pin_memory=pin_memory
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

    df_output = tiles_df.join(df_scores)
    df_output.columns = df_output.columns.astype(str)
    df_output.index.name = "address"

    logger.info(df_output)

    with ofs.open(output_file, "wb") as of:
        df_output.to_parquet(of)

    # Save our properties and params
    properties = {
        "tiles_url": ofs.unstrip_protocol(output_file),
        "total_tiles": len(df_output),
        "available_labels": list(df_output.columns),
    }

    return properties


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
