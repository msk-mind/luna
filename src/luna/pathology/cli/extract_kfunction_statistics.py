# General imports
from pathlib import Path

import fire
import fsspec
import numpy as np
import pandas as pd
from dask.distributed import Client, progress
from loguru import logger
from tqdm.contrib.itertools import product

from luna.common.dask import get_or_create_dask_client
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.common.utils import coord_to_address
from luna.pathology.spatial.stats import Kfunction


@timed
@save_metadata
def cli(
    input_cell_objects_urlpath: str = "???",
    tile_size: int = "???",  # type: ignore
    intensity_label: str = "???",
    tile_stride: int = "???",  # type: ignore
    radius: float = "???",  # type: ignore
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Run k function using a sliding window approach, where the k-function is computed locally in a smaller window, and aggregated across the entire slide.

    Args:
        input_cell_objects_urlpath (str): url/path to cell objects (.csv)
        tile_size (int): size of tiles to use (at the requested magnification)
        tile_stride (int): spacing between tiles
        intensity_label (str): Columns of cell object to use for intensity calculations (for I-K function - spatial + some scalar value clustering)
        radius (float):  the radius to consider
        output_urlpath (str): output URL/path prefix
        storage_options (dict): storage options for reading the cell objects

    Returns:
        pd.DataFrame: metadata about function call
    """
    config = get_config(vars())
    df_stats = extract_kfunction(
        config["input_cell_objects_urlpath"],
        config["tile_size"],
        config["intensity_label"],
        config["tile_stride"],
        config["radius"],
        config["storage_options"],
    )
    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_tile_header = Path(output_urlpath_prefix) / (
        str(Path(config["input_cell_objects_urlpath"]).stem)
        + "_kfunction_supertiles.parquet"
    )
    with fs.open(output_tile_header, "wb") as of:
        df_stats.to_parquet(of)

    properties = {
        "slide_tiles": output_tile_header,
    }

    return properties


def extract_kfunction(
    input_cell_objects_urlpath: str,
    tile_size: int,
    intensity_label: str,
    tile_stride: int,
    radius: float,
    storage_options: dict = {},
):
    """Run k function using a sliding window approach, where the k-function is computed locally in a smaller window, and aggregated across the entire slide.

    Args:
        input_cell_objects (str): URL/path to cell objects (.csv)
        tile_size (int): size of tiles to use (at the requested magnification)
        intensity_label (str): Columns of cell object to use for intensity calculations (for I-K function - spatial + some scalar value clustering)
        tile_stride (int): spacing between tiles
        radius (float):  the radius to consider
        storage_options (dict): storage options for reading the cell objects

    Returns:
        dict: metadata about function call
    """
    client = get_or_create_dask_client()
    df = pd.read_parquet(input_cell_objects_urlpath, storage_options=storage_options)

    l_address = []
    l_k_function_futures = []
    l_x_coord = []
    l_y_coord = []

    feature_name = (
        f"ikfunction_r{radius}_stain{intensity_label.replace(' ','_').replace(':','')}"
    )

    coords = product(
        range(int(df["x_coord"].min()), int(df["x_coord"].max()), tile_stride),
        range(int(df["y_coord"].min()), int(df["y_coord"].max()), tile_stride),
    )

    logger.info("Submitting tasks...")
    for x, y in coords:
        df_tile = df.query(
            f"x_coord >= {x} and x_coord <= {x+tile_size} and y_coord >={y} and y_coord <= {y+tile_size}"
        )

        if len(df_tile) < 3:
            continue

        future = client.submit(
            Kfunction,
            df_tile[["x_coord", "y_coord"]],
            df_tile[["x_coord", "y_coord"]],
            intensity=np.array(df_tile[intensity_label]),
            radius=radius,
            count=True,
        )

        l_address.append(coord_to_address((x, y), 0))
        l_k_function_futures.append(future)
        l_x_coord.append(x)
        l_y_coord.append(y)
    logger.info("Waiting for all tasks to complete...")
    progress(l_k_function_futures)
    l_k_function = client.gather(l_k_function_futures)

    df_stats = pd.DataFrame(
        {
            "address": l_address,
            "x_coord": l_x_coord,
            "y_coord": l_y_coord,
            "results": l_k_function,
        }
    ).set_index("address")
    df_stats.loc[:, "xy_extent"] = tile_size
    df_stats.loc[:, "tile_size"] = tile_size  # Same, 1 to 1
    df_stats.loc[:, "tile_units"] = "um"  # Same, 1 to 1

    df_stats[feature_name] = df_stats["results"].apply(lambda x: x["intensity"])
    df_stats[feature_name + "_norm"] = (
        df_stats[feature_name] / df_stats[feature_name].max()
    )

    df_stats = df_stats.drop(columns=["results"]).dropna()

    logger.info("Generated k-function feature data:")
    logger.info(df_stats)

    return df_stats


if __name__ == "__main__":
    Client()
    fire.Fire(cli)
