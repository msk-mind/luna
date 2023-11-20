# General imports
from pathlib import Path

import dask.bag as db
import fire
import fsspec
import h5py
import pandas as pd
from dask.diagnostics import ProgressBar
from loguru import logger
from pandera.typing import DataFrame
from tiffslide import TiffSlide

from luna.common.dask import configure_dask_client, get_or_create_dask_client
from luna.common.models import SlideSchema
from luna.common.utils import get_config, local_cache_urlpath, save_metadata, timed
from luna.pathology.common.utils import get_array_from_tile


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tiles_urlpath: str = "???",
    batch_size: int = 2000,
    output_urlpath: str = ".",
    force: bool = False,
    storage_options: dict = {},
    output_storage_options: dict = {},
    dask_options: dict = {},
    local_config: str = "",
):
    """Saves tiles to disk

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): url/path to tile manifest (.parquet)
        batch_size (int): size in batch dimension to chuck jobs
        output_urlpath (str): output url/path prefix
        force (bool): overwrite outputs if they exist
        storage_options (dict): storage options to reading functions
        output_storage_options (dict): storage options to writing functions
        dask_options (dict): dask options
        local_config (str): url/path to local config yaml file

    Returns:
        dict: metadata about function call
    """
    config = get_config(vars())

    configure_dask_client(**config["dask_options"])

    properties = _save_tiles(
        config["tiles_urlpath"],
        config["slide_urlpath"],
        config["output_urlpath"],
        config["force"],
        config["batch_size"],
        config["storage_options"],
        config["output_storage_options"],
    )

    return properties


def save_tiles(
    slide_manifest: DataFrame[SlideSchema],
    output_urlpath: str,
    force: bool = True,
    batch_size: int = 2000,
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> DataFrame[SlideSchema]:
    """Saves tiles to disk

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        slide_manifest (DataFrame[SlideSchema]): slide manifest from slide_etl
        output_urlpath (str): output url/path prefix
        force (bool): overwrite outputs if they exist
        batch_size (int): size in batch dimension to chuck jobs
        storage_options (dict): storage options to reading functions
        output_storage_options (dict): storage options to writing functions

    Returns:
        DataFrame[SlideSchema]: slide manifest
    """
    client = get_or_create_dask_client()

    if "tiles_url" not in slide_manifest.columns:
        raise ValueError("Generate tiles first")

    output_filesystem, output_path_prefix = fsspec.core.url_to_fs(
        output_urlpath, **output_storage_options
    )

    if not output_filesystem.exists(output_urlpath):
        output_filesystem.mkdir(output_urlpath)

    futures = []
    for slide in slide_manifest.itertuples(name="Slide"):
        future = client.submit(
            _save_tiles,
            slide.tiles_url,
            slide.url,
            output_urlpath,
            force,
            batch_size,
            storage_options,
            output_storage_options,
        )
        futures.append(future)

    results = client.gather(futures)
    return slide_manifest.assign(tiles_url=[x["tiles_url"] for x in results])


def _save_tiles(
    tiles_urlpath: str,
    slide_urlpath: str,
    output_urlpath: str,
    force: bool,
    batch_size: int = 2000,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    slide_id = Path(slide_urlpath).stem
    ofs, output_urlpath_prefix = fsspec.core.url_to_fs(
        output_urlpath, **output_storage_options
    )

    output_h5_path = str(Path(output_urlpath_prefix) / f"{slide_id}.tiles.h5")
    output_h5_url = ofs.unstrip_protocol(output_h5_path)

    output_tiles_path = str(Path(output_urlpath_prefix) / f"{slide_id}.tiles.parquet")
    output_tiles_url = ofs.unstrip_protocol(output_tiles_path)

    if ofs.exists(output_tiles_path) and ofs.exists(output_h5_path):
        logger.info(f"outputs already exist: {output_h5_url}, {output_tiles_url}")
        return
    tiles_df = __save_tiles(
        tiles_urlpath,
        slide_urlpath,
        output_h5_path,
        batch_size,
        storage_options,
        output_storage_options,
    )

    tiles_df["tile_store"] = output_h5_url
    logger.info(tiles_df)
    with ofs.open(output_tiles_path, "wb") as of:
        tiles_df.to_parquet(of)

    properties = {
        "tiles_url": output_tiles_url,  # "Tiles" are the metadata that describe them
        "feature_data": output_h5_url,  # Tiles can act like feature data
        "total_tiles": len(tiles_df),
    }

    return properties


@local_cache_urlpath(
    file_key_write_mode={
        "slide_urlpath": "r",
        "output_h5_path": "w",
    },
)
def __save_tiles(
    tiles_urlpath: str,
    slide_urlpath: str,
    output_h5_path: str,
    batch_size: int = 2000,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Saves tiles to disk

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        tiles_urlpath (str): tile manifest
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_urlpath (str): output url/path
        batch_size (int): size in batch dimension to chuck jobs
        output_storage_options (dict): storage options to writing functions

    Returns:
        dict: metadata about function call
    """

    tiles_df = pd.read_parquet(tiles_urlpath, storage_options=storage_options)

    get_or_create_dask_client()

    def f_many(iterator):
        with TiffSlide(slide_urlpath) as slide:
            return [(x.address, get_array_from_tile(x, slide=slide)) for x in iterator]

    chunks = db.from_sequence(
        tiles_df.itertuples(name="Tile"), partition_size=batch_size
    )

    ProgressBar().register()
    results = chunks.map_partitions(f_many)
    with h5py.File(output_h5_path, "w") as hfile:
        for result in results.compute():
            address, tile_arr = result
            hfile.create_dataset(address, data=tile_arr)

    return tiles_df


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
