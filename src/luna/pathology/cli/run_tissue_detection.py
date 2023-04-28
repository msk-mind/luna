# General imports
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import fire  # type: ignore
import fsspec  # type: ignore
import numpy as np
import pandas as pd
from dask.distributed import Client, get_client, progress
from fsspec import open  # type: ignore
from loguru import logger
from multimethod import multimethod
from pandera.typing import DataFrame
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore
from tiffslide import TiffSlide

from luna.common.models import SlideSchema, Tile
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.cli.generate_tiles import generate_tiles
from luna.pathology.common.utils import (
    get_array_from_tile,
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    get_stain_vectors_macenko,
    pull_stain_channel,
)


def compute_otsu_score(
    tile: Tile, otsu_threshold: float, slide_url: str, storage_options: dict
) -> float:
    """
    Return otsu score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
        otsu_threshold (float): otsu threshold value
    """
    with open(slide_url, **storage_options) as f:
        tile_arr = get_array_from_tile(tile, TiffSlide(f), size=(10, 10))
        score = np.mean((rgb2gray(tile_arr) < otsu_threshold).astype(int))
    return score


def get_purple_score(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    score = np.mean((r > (g + 10)) & (b > (g + 10)))
    return score


def compute_purple_score(tile: Tile, slide_url: str, storage_options: dict) -> float:
    """
    Return purple score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
    """
    with open(slide_url, **storage_options) as f:
        tile_arr = get_array_from_tile(tile, TiffSlide(f), size=(10, 10))
    return get_purple_score(tile_arr)


def compute_stain_score(
    tile: Tile,
    slide_url: str,
    vectors,
    channel,
    stain_threshold: float,
    storage_options: dict,
) -> np.floating:
    """
    Returns stain score for the tile
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
        vectors (np.ndarray): stain vectors
        channel (int): stain channel
        stain_threshold (float): stain threshold value
    """

    with open(slide_url, **storage_options) as f:
        tile_arr = get_array_from_tile(tile, TiffSlide(f), size=(10, 10))
    stain = pull_stain_channel(tile_arr, vectors=vectors, channel=channel)
    score = np.mean(stain > stain_threshold)
    return score


@timed
@save_metadata
def cli(
    slide: str = "???",
    filter_query: str = "???",
    tile_size: int = "???",
    requested_magnification: Optional[int] = None,
    output_url: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
) -> dict:
    """Run simple/deterministic tissue detection algorithms based on a filter query, to reduce tiles to those (likely) to contain actual tissue
    Args:
        slide (str): url to slide image (virtual slide formats compatible with pyvips, .svs, .tif, .scn, ...)
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (Optional[int]): Magnification scale at which to perform computation
        output_url (str): Output url prefix
        num_cores (int): Number of cores to use for CPU parallelization
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config file
    Returns:
        dict: metadata about cli function call

    """
    config = get_config(vars())

    output_filesystem, output_urlpath = fsspec.core.url_to_fs(
        config["output_url"], **config["output_storage_options"]
    )

    slide_id = Path(urlparse(config["slide"]).path).stem
    output_header_file = Path(output_urlpath) / f"{slide_id}-filtered.tiles.parquet"

    df = detect_tissue(
        config["slide"],
        config["tile_size"],
        config["requested_magnification"],
        config["filter_query"],
        config["storage_options"],
        config["output_url"],
        config["output_storage_options"],
    )
    with open(output_header_file, "wb", **config["output_storage_options"]) as of:
        print(f"saving to {output_header_file}")
        df.to_parquet(of)
        properties = {
            "tiles_manifest": output_header_file,
            "total_tiles": len(df),
        }

    return properties


@multimethod
def detect_tissue(
    slide_manifest: DataFrame,
    tile_size: int,
    requested_magnification: Optional[int] = None,
    filter_query: str = "",
    storage_options: dict = {},
    output_url_prefix: str = "",
    output_storage_options: dict = {},
):
    slide_manifest = SlideSchema(slide_manifest)
    dfs = []
    for row in slide_manifest.itertuples():
        df = __detect_tissue(
            row.url,
            tile_size,
            requested_magnification,
            filter_query,
            storage_options,
            output_url_prefix,
            output_storage_options,
        )
        df["id"] = row.id
        dfs.append(df)
    tiles = pd.concat(dfs)
    return tiles.merge(slide_manifest, on="id")


# this doesn't work:
#    client = get_client()
#    futures = {
#            row.id: client.submit(detect_tissue,
#                                  row.url,
#                                  tile_size,
#                                  requested_magnification,
#                                  filter_query,
#                                  storage_options,
#                                  output_url_prefix,
#                                  output_storage_options,
#                                  ) for row in slide_manifest.itertuples()
#            }
#    progress(futures)
#    results = client.gather(futures)
#    for k, v in results.items():
#        v['id'] = str(k)
#    tiles = pd.concat(results)
#    return tiles.merge(slide_manifest, on='id')


@multimethod
def detect_tissue(
    slide_urls: Union[str, list[str]],
    tile_size: int,
    requested_magnification: Optional[int] = None,
    filter_query: str = "",
    storage_options: dict = {},
    output_url_prefix: str = "",
    output_storage_options: dict = {},
) -> pd.DataFrame:
    if type(slide_urls) == str:
        slide_urls = [slide_urls]
    dfs = []
    for slide_url in slide_urls:
        df = __detect_tissue(
            slide_url,
            tile_size,
            requested_magnification,
            filter_query,
            storage_options,
            output_url_prefix,
            output_storage_options,
        )
        o = urlparse(slide_url)
        df["id"] = Path(o.path).stem
        dfs.append(df)
    return pd.concat(dfs)


def __detect_tissue(
    slide_url: str,
    tile_size: int,
    requested_magnification: Optional[int] = None,
    filter_query: str = "",
    storage_options: dict = {},
    output_url_prefix: str = "",
    output_storage_options: dict = {},
) -> pd.DataFrame:
    """Run simple/deterministic tissue detection algorithms based on a filter query, to reduce tiles to those (likely) to contain actual tissue
    Args:
        slide_url (str): slide url
        tile_size (int): size of tiles to use (at the requested magnification)
        requested_magnification (Optional[int]): Magnification scale at which to perform computation
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        storage_options (dict): storage options to pass to reading functions
        output_url_prefix (str): output url prefix
        output_storage_options (dict): output storage optoins
    Returns:
        pd.DataFrame
    """

    client = get_client()

    tiles_df = generate_tiles(
        slide_url, tile_size, storage_options, requested_magnification
    )

    with open(slide_url, "rb", **storage_options) as f:
        slide = TiffSlide(f)
        logger.info(f"Slide dimensions {slide.dimensions}")
        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=requested_magnification
        )
        logger.info(f"Thumbnail scale factor: {to_mag_scale_factor}")
        # Original thumbnail
        sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)
        logger.info(f"Sample array size: {sample_arr.shape}")

    if output_url_prefix:
        with open(
            output_url_prefix + "/sample_arr.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(sample_arr).save(f, format="png")

    # Enhance to drive stain apart from shadows, pushes darks from colors
    enhanced_sample_img = ImageEnhance.Contrast(
        ImageEnhance.Color(Image.fromarray(sample_arr)).enhance(10)
    ).enhance(10)
    if output_url_prefix:
        with open(
            output_url_prefix + "/enhanced_sample_arr.png",
            "wb",
            **output_storage_options,
        ) as f:
            enhanced_sample_img.save(f, format="png")

    # Look at HSV space
    hsv_sample_arr = np.array(enhanced_sample_img.convert("HSV"))
    if output_url_prefix:
        with open(
            output_url_prefix + "/hsv_sample_arr.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(np.array(hsv_sample_arr)).save(f, "png")

    # Look at max of saturation and value
    hsv_max_sample_arr = np.max(hsv_sample_arr[:, :, 1:3], axis=2)
    if output_url_prefix:
        with open(
            output_url_prefix + "/hsv_max_sample_arr.png",
            "wb",
            **output_storage_options,
        ) as f:
            Image.fromarray(hsv_max_sample_arr).save(f, "png")

    # Get shadow mask and filter it out before other estimations
    shadow_mask = np.where(np.max(hsv_sample_arr, axis=2) < 10, 255, 0).astype(np.uint8)
    if output_url_prefix:
        with open(
            output_url_prefix + "/shadow_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(shadow_mask).save(f, "png")

    # Filter out shadow/dust/etc
    sample_arr_filtered = np.where(
        np.expand_dims(shadow_mask, 2) == 0, sample_arr, np.full(sample_arr.shape, 255)
    ).astype(np.uint8)
    if output_url_prefix:
        with open(
            output_url_prefix + "/sample_arr_filtered.png",
            "wb",
            **output_storage_options,
        ) as f:
            Image.fromarray(sample_arr_filtered).save(f, "png")

    # Get otsu threshold
    threshold = threshold_otsu(rgb2gray(sample_arr_filtered))

    # Get stain vectors
    stain_vectors = get_stain_vectors_macenko(sample_arr_filtered)

    # Get stain background thresholds
    threshold_stain0 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=0
        ).flatten()
    )
    threshold_stain1 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=1
        ).flatten()
    )

    # Get the otsu mask
    if output_url_prefix:
        otsu_mask = np.where(rgb2gray(sample_arr_filtered) < threshold, 255, 0).astype(
            np.uint8
        )
        with open(
            output_url_prefix + "/otsu_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(otsu_mask).save(f, "png")

    if output_url_prefix:
        # Get stain thumnail image
        deconv_sample_arr = pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors
        )
        with open(output_url_prefix + "/deconv_sample_arr.png", "wb") as f:
            Image.fromarray(deconv_sample_arr).save(f, "png", **output_storage_options)

        # Get the stain masks
        stain0_mask = np.where(
            deconv_sample_arr[..., 0] > threshold_stain0, 255, 0
        ).astype(np.uint8)
        stain1_mask = np.where(
            deconv_sample_arr[..., 1] > threshold_stain1, 255, 0
        ).astype(np.uint8)
        with open(
            output_url_prefix + "/stain0_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(stain0_mask).save(f, "png")
        with open(
            output_url_prefix + "/stain1_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(stain1_mask).save(f, "png")

    if filter_query:
        if "otsu_score" in filter_query:
            logger.info(f"Starting otsu thresholding, threshold={threshold}")
            otsu_score_futures = [
                client.submit(
                    compute_otsu_score,
                    row,
                    slide_url=slide_url,
                    storage_options=storage_options,
                    otsu_threshold=threshold,
                )
                for row in tiles_df.itertuples(name="Tile")
            ]
            progress(otsu_score_futures)
            tiles_df["otsu_score"] = client.gather(otsu_score_futures)
        if "purple_score" in filter_query:
            logger.info("Starting purple scoring")
            purple_futures = [
                client.submit(
                    compute_purple_score,
                    row,
                    slide_url=slide_url,
                    storage_options=storage_options,
                )
                for row in tiles_df.itertuples(name="Tile")
            ]
            progress(purple_futures)
            tiles_df["purple_score"] = client.gather(purple_futures)
        if "stain0_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=0, threshold={threshold_stain0}"
            )
            stain0_futures = [
                client.submit(
                    compute_stain_score,
                    row,
                    slide_url=slide_url,
                    storage_options=storage_options,
                    vectors=stain_vectors,
                    channel=0,
                    stain_threshold=threshold_stain0,
                )
                for row in tiles_df.itertuples(name="Tile")
            ]
            progress(stain0_futures)
            tiles_df["stain0_score"] = client.gather(stain0_futures)
        if "stain1_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=1, threshold={threshold_stain1}"
            )
            stain1_futures = [
                client.submit(
                    compute_stain_score,
                    row,
                    slide_url=slide_url,
                    storage_options=storage_options,
                    vectors=stain_vectors,
                    channel=1,
                    stain_threshold=threshold_stain1,
                )
                for row in tiles_df.itertuples(name="Tile")
            ]
            progress(stain1_futures)
            tiles_df["stain1_score"] = client.gather(stain1_futures)

        logger.info(f"Filtering based on query: {filter_query}")
        tiles_df = tiles_df.query(filter_query)

    logger.info(tiles_df)

    return tiles_df


if __name__ == "__main__":
    client = Client()
    fire.Fire(cli)
