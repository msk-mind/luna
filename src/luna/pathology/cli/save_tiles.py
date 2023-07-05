# General imports
from pathlib import Path
from typing import Optional

import fire
import fsspec
import h5py
from dask.distributed import as_completed, progress
from loguru import logger

from luna.common.dask import get_or_create_dask_client
from luna.common.utils import get_config, grouper, save_metadata, timed
from luna.pathology.cli.generate_tiles import generate_tiles
from luna.pathology.common.utils import get_array_from_tile


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tile_size: int = "???",  # type: ignore
    requested_magnification: Optional[int] = None,
    batch_size: int = 1000,
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Saves tiles to disk

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation
        output_urlpath (str): output url/path prefix
        batch_size (int): size in batch dimension to chuck jobs
        storage_options (dict): storage options to reading functions
        output_storage_options (dict): storage options to writing functions
        local_config (str): url/path to local config yaml file

    Returns:
        dict: metadata about function call
    """
    config = get_config(vars())

    df = save_tiles(
        config["slide_urlpath"],
        config["tile_size"],
        config["output_urlpath"],
        config["batch_size"],
        config["requested_magnification"],
        config["storage_options"],
        config["output_storage_options"],
    )

    logger.info(df)
    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    slide_id = Path(config["slide_urlpath"]).stem
    output_header_file = Path(output_urlpath_prefix) / f"{slide_id}.tiles.parquet"
    with fs.open(output_header_file, "wb") as of:
        df.to_parquet(of)

    properties = {
        "slide_tiles": output_header_file,  # "Tiles" are the metadata that describe them
        "feature_data": output_header_file,  # Tiles can act like feature data
        "total_tiles": len(df),
    }

    return properties


def save_tiles(
    slide_urlpath: str,
    tile_size: int,
    output_urlpath: str,
    batch_size: int = 1000,
    requested_magnification: Optional[int] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
):
    """Saves tiles to disk

    Tiles addresses and arrays are saved as key-value pairs in (tiles.h5),
    and the corresponding manifest/header file (tiles.parquet) is also generated

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (float): Magnification scale at which to perform computation
        output_urlpath (str): output url/path prefix
        batch_size (int): size in batch dimension to chuck jobs
        storage_options (dict): storage options to reading functions
        output_storage_options (dict): storage options to writing functions

    Returns:
        dict: metadata about function call
    """
    client = get_or_create_dask_client()
    slide_id = Path(slide_urlpath).stem
    df = generate_tiles(
        slide_urlpath, tile_size, storage_options, requested_magnification
    )

    logger.info(f"Now generating tiles with batch_size={batch_size}!")

    # save address:tile arrays key:value pair in hdf5
    def f_many(iterator):
        return [
            (
                x.address,
                get_array_from_tile(x, slide_urlpath, storage_options=storage_options),
            )
            for x in iterator
        ]

    chunks = grouper(df.itertuples(name="Tile"), batch_size)

    futures = client.map(f_many, chunks)
    progress(futures)

    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        output_urlpath, **output_storage_options
    )
    output_hdf_file = Path(output_urlpath_prefix) / f"{slide_id}.tiles.h5"
    progress(futures)

    simplecache_fs = fsspec.filesystem("simplecache", target_protocol=fs.protocol)

    with simplecache_fs.open(output_hdf_file, "wb") as of:
        with h5py.File(of, "x") as hfile:
            for future in as_completed(futures):
                for result in future.result():
                    address, tile_arr = result
                    hfile.create_dataset(address, data=tile_arr)

    df["tile_store"] = fs.unstrip_protocol(str(output_hdf_file))
    return df


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
