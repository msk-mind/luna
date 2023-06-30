# General imports
from pathlib import Path
from urllib.parse import urlparse

import fire
import fsspec
import pandas as pd
from loguru import logger

import luna.common.stats
from luna.common.utils import get_config, save_metadata, timed


@timed
@save_metadata
def cli(
    tiles_urlpath: str = "???",
    output_urlpath: str = "???",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Extracts statistics over tiles

    Args:
        tiles_urlpath (str): Tiles parquet file for slide(s). Absolute or relative filepath. Prefix with protocol to read from alternative filesystems
        output_urlpath (str): Output prefix. Absolute or relative filepath. Prefix with protocol to write to alternative filesystems
        storage_options (dict): extra options that make sense for reading from a particular storage connection
        output_storage_options (dict): extra options that make sense for writing to a particular storage connection
        local_config (str): local config yaml file

    """
    config = get_config(vars())

    df_feature_data = extract_tile_statistics(
        config["tiles_urlpath"],
        config["storage_options"],
    )

    fs, output_path_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    o = urlparse(config["tiles_urlpath"])
    id = Path(o.path).stem

    output_feature_file = Path(output_path_prefix) / f"{id}_tile_stats.parquet"

    logger.info(df_feature_data)
    with fs.open(output_feature_file, "wb") as f:
        df_feature_data.to_parquet(f)

    properties = {"feature_data": output_feature_file}

    return properties


def extract_tile_statistics(
    tiles_urlpath: str,
    storage_options: dict,
):
    """Extracts statistics over tiles

    Args:
        tiles_urlpath (str): Tiles parquet file for slide(s). Absolute or relative filepath. Prefix with protocol to read from alternative filesystems
        output_urlpath (str): Output prefix. Absolute or relative filepath. Prefix with protocol to write to alternative filesystems
        storage_options (dict): extra options that make sense for reading from a particular storage connection
        output_storage_options (dict): extra options that make sense for writing to a particular storage connection

    Returns:
        pd.DataFrame: metadata about function call
    """

    df = (
        pd.read_parquet(tiles_urlpath, storage_options=storage_options)
        .reset_index()
        .set_index("address")
        .drop(
            columns=["x_coord", "y_coord", "tile_size", "xy_extent", "tile_units"],
            errors="ignore",
        )
    )
    print(df.columns)

    dict_feature_data = {}

    for col in df.columns:
        dict_feature_data.update(
            luna.common.stats.compute_stats_1d(pd.to_numeric(df[col]), col)
        )

    df_feature_data = pd.DataFrame([dict_feature_data])

    return df_feature_data


if __name__ == "__main__":
    fire.Fire(cli)
