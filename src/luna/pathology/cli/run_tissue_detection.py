# General imports
from functools import partial
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import fire  # type: ignore
import fsspec  # type: ignore
import numpy as np
import pandas as pd
from dask.distributed import Client, progress
from fsspec import open  # type: ignore
from loguru import logger
from multimethod import multimethod
from pandera.typing import DataFrame
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore
from tiffslide import TiffSlide

from luna.common.dask import get_or_create_dask_client, configure_dask_client
from luna.common.models import SlideSchema, Tile
from luna.common.utils import get_config, grouper, local_cache_urlpath, save_metadata, timed
from luna.pathology.cli.generate_tiles import generate_tiles
from luna.pathology.common.utils import (
    get_array_from_tile,
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    get_stain_vectors_macenko,
    pull_stain_channel,
)


def compute_otsu_score(tile: Tile, slide: TiffSlide, otsu_threshold: float) -> float:
    """
    Return otsu score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_urlpath (str): path to slide
        otsu_threshold (float): otsu threshold value
    """
    tile_arr = get_array_from_tile(tile, slide, 10)
    score = np.mean((rgb2gray(tile_arr) < otsu_threshold).astype(int))
    return score


def get_purple_score(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    score = np.mean((r > (g + 10)) & (b > (g + 10)))
    return score


def compute_purple_score(
    tile: Tile,
    slide: TiffSlide,
) -> float:
    """
    Return purple score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
    """
    tile_arr = get_array_from_tile(tile, slide, 10)
    return get_purple_score(tile_arr)


def compute_stain_score(
    tile: Tile,
    slide: TiffSlide,
    vectors,
    channel,
    stain_threshold: float,
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
    tile_arr = get_array_from_tile(tile, slide, 10)
    stain = pull_stain_channel(tile_arr, vectors=vectors, channel=channel)
    score = np.mean(stain > stain_threshold)
    return score


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tiles_urlpath: str = "",
    filter_query: str = "???",
    tile_size: Optional[int] = None,
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    batch_size: int = 2000,
    output_urlpath: str = ".",
    dask_options: dict = {},
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
) -> dict:
    """Run simple/deterministic tissue detection algorithms based on a filter query, to reduce tiles to those (likely) to contain actual tissue
    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with pyvips, .svs, .tif, .scn, ...)
        tiles_urlpath (str): url/path to tiles manifest (parquet)
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        tile_size (int): size of tiles to use (at the requested magnification)
        thumbnail_magnification (Optional[int]): Magnification scale at which to perform computation
        output_urlpath (str): Output url/path prefix
        dask_options (dict): dask options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config file
    Returns:
        dict: metadata about cli function call

    """
    config = get_config(vars())

    configure_dask_client(**config['dask_options'])

    if not config["tile_size"] and not config["tiles_urlpath"]:
        raise fire.core.FireError("Specify either tiles_urlpath or tile_size")

    output_filesystem, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    slide_id = Path(urlparse(config["slide_urlpath"]).path).stem
    output_header_file = (
        Path(output_urlpath_prefix) / f"{slide_id}.tiles.parquet"
    )

    df = detect_tissue(
        config["slide_urlpath"],
        config["tiles_urlpath"],
        config["tile_size"],
        config["thumbnail_magnification"],
        config["tile_magnification"],
        config["filter_query"],
        config["batch_size"],
        config["storage_options"],
        config["output_urlpath"],
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
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    filter_query: str = "",
    batch_size: int = 2000,
    storage_options: dict = {},
    output_urlpath_prefix: str = "",
    output_storage_options: dict = {},
):
    slide_manifest = SlideSchema(slide_manifest)
    dfs = []
    for row in slide_manifest.itertuples():
        df = detect_tissue(
            row.url,
            tile_size,
            thumbnail_magnification,
            tile_magnification,
            filter_query,
            batch_size,
            storage_options,
            output_urlpath_prefix,
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
#                                  thumbnail_magnification,
#                                  filter_query,
#                                  storage_options,
#                                  output_urlpath_prefix,
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
    slide_urlpaths: Union[str, list[str]],
    tile_size: int,
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    filter_query: str = "",
    batch_size: int = 2000,
    storage_options: dict = {},
    output_urlpath_prefix: str = "",
    output_storage_options: dict = {},
) -> pd.DataFrame:
    if type(slide_urlpaths) == str:
        slide_urlpaths = [slide_urlpaths]
    dfs = []
    for slide_urlpath in slide_urlpaths:
        df = detect_tissue(
            slide_urlpath,
            "",
            tile_size,
            thumbnail_magnification,
            tile_magnification,
            filter_query,
            batch_size,
            storage_options,
            output_urlpath_prefix,
            output_storage_options,
        )
        o = urlparse(slide_urlpath)
        df["id"] = Path(o.path).stem
        dfs.append(df)
    return pd.concat(dfs)


@multimethod
@local_cache_urlpath(
    file_key_write_mode={
        "slide_urlpath": "r",
    },
)
def detect_tissue(
    slide_urlpath: str,
    tiles_urlpath: str = "",
    tile_size: Optional[int] = None,
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    filter_query: str = "",
    batch_size: int = 2000,
    storage_options: dict = {},
    output_urlpath_prefix: str = "",
    output_storage_options: dict = {},
) -> pd.DataFrame:
    """Run simple/deterministic tissue detection algorithms based on a filter query, to reduce tiles to those (likely) to contain actual tissue
    Args:
        slide_urlpath (str): slide url/path
        tile_size (int): size of tiles to use (at the requested magnification)
        thumbnail_magnification (Optional[int]): Magnification scale at which to perform computation
        filter_query (str): pandas query by which to filter tiles based on their various tissue detection scores
        storage_options (dict): storage options to pass to reading functions
        output_urlpath_prefix (str): output url/path prefix
        output_storage_options (dict): output storage optoins
    Returns:
        pd.DataFrame
    """

    client = get_or_create_dask_client()

    if tiles_urlpath:
        with open(tiles_urlpath, **storage_options) as of:
            tiles_df = pd.read_parquet(of)
    elif type(tile_size) == int:
        tiles_df = generate_tiles(slide_urlpath, tile_size, storage_options, tile_magnification)
    else:
        raise RuntimeError("Specify tile_size or tile_urlpath")

    with TiffSlide(slide_urlpath) as slide:
        logger.info(f"Slide dimensions {slide.dimensions}")
        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=thumbnail_magnification
        )
        logger.info(f"Thumbnail scale factor: {to_mag_scale_factor}")
        # Original thumbnail
        sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)
        logger.info(f"Sample array size: {sample_arr.shape}")

    with TiffSlide(slide_urlpath) as slide:
        logger.info(f"Slide dimensions {slide.dimensions}")
        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=thumbnail_magnification
        )
        logger.info(f"Thumbnail scale factor: {to_mag_scale_factor}")
        # Original thumbnail
        sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)
        logger.info(f"Sample array size: {sample_arr.shape}")

    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/sample_arr.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(sample_arr).save(f, format="png")

    logger.info("Enhancing image...")
    enhanced_sample_img = ImageEnhance.Contrast(
        ImageEnhance.Color(Image.fromarray(sample_arr)).enhance(10)
    ).enhance(10)
    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/enhanced_sample_arr.png",
            "wb",
            **output_storage_options,
        ) as f:
            enhanced_sample_img.save(f, format="png")

    logger.info("HSV space conversion...")
    hsv_sample_arr = np.array(enhanced_sample_img.convert("HSV"))
    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/hsv_sample_arr.png",
            "wb",
            **output_storage_options,
        ) as f:
            Image.fromarray(np.array(hsv_sample_arr)).save(f, "png")

    logger.info("Calculating max saturation...")
    hsv_max_sample_arr = np.max(hsv_sample_arr[:, :, 1:3], axis=2)
    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/hsv_max_sample_arr.png",
            "wb",
            **output_storage_options,
        ) as f:
            Image.fromarray(hsv_max_sample_arr).save(f, "png")

    logger.info("Calculate and filter shadow mask...")
    shadow_mask = np.where(np.max(hsv_sample_arr, axis=2) < 10, 255, 0).astype(np.uint8)
    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/shadow_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(shadow_mask).save(f, "png")

    logger.info("Filter out shadow/dust/etc...")
    sample_arr_filtered = np.where(
        np.expand_dims(shadow_mask, 2) == 0, sample_arr, np.full(sample_arr.shape, 255)
    ).astype(np.uint8)
    if output_urlpath_prefix:
        with open(
            output_urlpath_prefix + "/sample_arr_filtered.png",
            "wb",
            **output_storage_options,
        ) as f:
            Image.fromarray(sample_arr_filtered).save(f, "png")

    logger.info("Calculating otsu threshold...")
    threshold = threshold_otsu(rgb2gray(sample_arr_filtered))

    logger.info("Calculating stain vectors...")
    stain_vectors = get_stain_vectors_macenko(sample_arr_filtered)

    logger.info("Calculating stain background thresholds...")
    logger.info("Channel 0")
    threshold_stain0 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=0
        ).flatten()
    )
    logger.info("Channel 1")
    threshold_stain1 = threshold_otsu(
        pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors, channel=1
        ).flatten()
    )

    # Get the otsu mask
    if output_urlpath_prefix:
        logger.info("Saving otsu mask")
        otsu_mask = np.where(rgb2gray(sample_arr_filtered) < threshold, 255, 0).astype(
            np.uint8
        )
        with open(
            output_urlpath_prefix + "/otsu_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(otsu_mask).save(f, "png")

    if output_urlpath_prefix:
        logger.info("Saving stain thumbnail")
        deconv_sample_arr = pull_stain_channel(
            sample_arr_filtered, vectors=stain_vectors
        )
        with open(output_urlpath_prefix + "/deconv_sample_arr.png", "wb") as f:
            Image.fromarray(deconv_sample_arr).save(f, "png", **output_storage_options)

        logger.info("Saving stain masks")
        stain0_mask = np.where(
            deconv_sample_arr[..., 0] > threshold_stain0, 255, 0
        ).astype(np.uint8)
        stain1_mask = np.where(
            deconv_sample_arr[..., 1] > threshold_stain1, 255, 0
        ).astype(np.uint8)
        with open(
            output_urlpath_prefix + "/stain0_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(stain0_mask).save(f, "png")
        with open(
            output_urlpath_prefix + "/stain1_mask.png", "wb", **output_storage_options
        ) as f:
            Image.fromarray(stain1_mask).save(f, "png")

    if filter_query:

        def f_many(iterator, tile_fn):
            with TiffSlide(slide_urlpath) as slide:
                return [tile_fn(tile=x, slide=slide) for x in iterator]

        if "otsu_score" in filter_query:
            logger.info(f"Starting otsu thresholding, threshold={threshold}")

            chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)
            otsu_tile_fn = partial(compute_otsu_score, otsu_threshold=threshold)

            futures = client.map(partial(f_many, tile_fn=otsu_tile_fn), chunks)
            progress(futures)
            tiles_df["otsu_score"] = np.concatenate(client.gather(futures))
        if "purple_score" in filter_query:
            logger.info("Starting purple scoring")
            chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)

            futures = client.map(partial(f_many, tile_fn=compute_purple_score), chunks)
            progress(futures)
            tiles_df["purple_score"] = np.concatenate(client.gather(futures))
        if "stain0_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=0, threshold={threshold_stain0}"
            )

            chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)
            stain_tile_fn = partial(
                compute_stain_score,
                vectors=stain_vectors,
                channel=0,
                stain_threshold=threshold_stain0,
            )

            futures = client.map(partial(f_many, tile_fn=stain_tile_fn), chunks)
            progress(futures)
            tiles_df["stain0_score"] = np.concatenate(client.gather(futures))
        if "stain1_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=1, threshold={threshold_stain1}"
            )
            chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)
            stain_tile_fn = partial(
                compute_stain_score,
                vectors=stain_vectors,
                channel=1,
                stain_threshold=threshold_stain1,
            )

            futures = client.map(partial(f_many, tile_fn=stain_tile_fn), chunks)
            progress(futures)
            tiles_df["stain1_score"] = np.concatenate(client.gather(futures))

        logger.info(f"Filtering based on query: {filter_query}")
        tiles_df = tiles_df.query(filter_query)

    logger.info(tiles_df)

    return tiles_df


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
