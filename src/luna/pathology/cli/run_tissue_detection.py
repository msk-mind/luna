# General imports
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import dask.bag as db
import fire  # type: ignore
import fsspec  # type: ignore
import numpy as np
import pandas as pd
from dask.distributed import progress
from loguru import logger
from pandera.typing import DataFrame
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray  # type: ignore
from skimage.filters import threshold_otsu  # type: ignore
from tiffslide import TiffSlide

from luna.common.dask import configure_dask_client, get_or_create_dask_client
from luna.common.models import Tile
from luna.common.utils import (
    get_config,
    local_cache_urlpath,
    make_temp_directory,
    save_metadata,
    timed,
)
from luna.pathology.cli.generate_tiles import __generate_tiles, generate_tiles
from luna.pathology.common.utils import (
    get_array_from_tile,
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    get_stain_vectors_macenko,
    pull_stain_channel,
)


def compute_otsu_score(tile: Tile, slide_path: str, otsu_threshold: float) -> float:
    """
    Return otsu score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_path (str): path to slide
        otsu_threshold (float): otsu threshold value
    """
    with TiffSlide(slide_path) as slide:
        tile_arr = get_array_from_tile(tile, slide, 10)
    score = np.mean((rgb2gray(tile_arr) < otsu_threshold).astype(int))
    return score


def get_purple_score(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    score = np.mean((r > (g + 10)) & (b > (g + 10)))
    return score


def compute_purple_score(
    tile: Tile,
    slide_path: str,
) -> float:
    """
    Return purple score for the tile.
    Args:
        row (pd.Series): row with tile metadata
        slide_url (str): path to slide
    """
    with TiffSlide(slide_path) as slide:
        tile_arr = get_array_from_tile(tile, slide, 10)
    return get_purple_score(tile_arr)


def compute_stain_score(
    tile: Tile,
    slide_path: str,
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
    with TiffSlide(slide_path) as slide:
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
        output_urlpath (str): Output url/path
        dask_options (dict): dask options
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): local config file
    Returns:
        dict: metadata about cli function call

    """
    config = get_config(vars())

    configure_dask_client(**config["dask_options"])

    if not config["tile_size"] and not config["tiles_urlpath"]:
        raise fire.core.FireError("Specify either tiles_urlpath or tile_size")

    slide_id = Path(config["slide_urlpath"]).stem

    tiles_urlpath = config["tiles_urlpath"]

    with make_temp_directory() as temp_dir:
        if not tiles_urlpath:
            result = __generate_tiles(
                config["slide_urlpath"],
                config["tile_size"],
                temp_dir,
                config["tile_magnification"],
                config["storage_options"],
            )
            tiles_urlpath = result["tiles_url"]

        properties = __detect_tissue(
            config["slide_urlpath"],
            tiles_urlpath,
            slide_id,
            config["thumbnail_magnification"],
            config["filter_query"],
            config["batch_size"],
            config["output_urlpath"],
            config["storage_options"],
            config["output_storage_options"],
        )

    return properties


def detect_tissue(
    slide_manifest: DataFrame,
    tile_size: Optional[int] = None,
    thumbnail_magnification: Optional[int] = None,
    tile_magnification: Optional[int] = None,
    filter_query: str = "",
    batch_size: int = 2000,
    storage_options: dict = {},
    output_urlpath: str = ".",
    output_storage_options: dict = {},
) -> pd.DataFrame:
    client = get_or_create_dask_client()

    with make_temp_directory() as temp_dir:
        if "tiles_url" not in slide_manifest.columns:
            slide_manifest = generate_tiles(
                slide_manifest,
                tile_size,
                temp_dir,
                tile_magnification,
                storage_options,
            )

        futures = []
        for slide in slide_manifest.itertuples(name="Slide"):
            future = client.submit(
                __detect_tissue,
                slide.url,
                slide.tiles_url,
                slide.id,
                thumbnail_magnification,
                filter_query,
                batch_size,
                output_urlpath,
                storage_options,
                output_storage_options,
            )
            futures.append(future)
        progress(futures)

        results = client.gather(futures)

        for idx, result in enumerate(results):
            slide_manifest.at[idx, "tiles_url"] = result["tiles_url"]

    return slide_manifest


@local_cache_urlpath(
    file_key_write_mode={
        "slide_urlpath": "r",
    }
)
def __detect_tissue(
    slide_urlpath: str,
    tiles_urlpath: str,
    slide_id: str,
    thumbnail_magnification: Optional[int] = None,
    filter_query: str = "",
    batch_size: int = 2000,
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
) -> Dict:
    output_filesystem, output_path = fsspec.core.url_to_fs(
        output_urlpath, **output_storage_options
    )

    tiles_output_path = str(Path(output_path) / f"{slide_id}.tiles.parquet")
    if output_filesystem.exists(tiles_output_path):
        logger.info(
            "Outputs already exist: {output_filesystem.unstrip_protocol(tiles_output_path)}"
        )
        return

    tiles_df = pd.read_parquet(tiles_urlpath, storage_options=storage_options)

    get_or_create_dask_client()

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

    with output_filesystem.open(Path(output_path) / "sample_arr.png", "wb") as f:
        Image.fromarray(sample_arr).save(f, format="png")

    logger.info("Enhancing image...")
    enhanced_sample_img = ImageEnhance.Contrast(
        ImageEnhance.Color(Image.fromarray(sample_arr)).enhance(10)
    ).enhance(10)
    with output_filesystem.open(
        Path(output_path) / "enhanced_sample_arr.png",
        "wb",
    ) as f:
        enhanced_sample_img.save(f, format="png")

    logger.info("HSV space conversion...")
    hsv_sample_arr = np.array(enhanced_sample_img.convert("HSV"))
    with output_filesystem.open(
        Path(output_path) / "hsv_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(np.array(hsv_sample_arr)).save(f, "png")

    logger.info("Calculating max saturation...")
    hsv_max_sample_arr = np.max(hsv_sample_arr[:, :, 1:3], axis=2)
    with output_filesystem.open(
        Path(output_path) / "hsv_max_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(hsv_max_sample_arr).save(f, "png")

    logger.info("Calculate and filter shadow mask...")
    shadow_mask = np.where(np.max(hsv_sample_arr, axis=2) < 10, 255, 0).astype(np.uint8)
    with output_filesystem.open(
        Path(output_path) / "shadow_mask.png",
        "wb",
    ) as f:
        Image.fromarray(shadow_mask).save(f, "png")

    logger.info("Filter out shadow/dust/etc...")
    sample_arr_filtered = np.where(
        np.expand_dims(shadow_mask, 2) == 0, sample_arr, np.full(sample_arr.shape, 255)
    ).astype(np.uint8)
    with output_filesystem.open(
        Path(output_path) / "sample_arr_filtered.png",
        "wb",
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
    logger.info("Saving otsu mask")
    otsu_mask = np.where(rgb2gray(sample_arr_filtered) < threshold, 255, 0).astype(
        np.uint8
    )
    with output_filesystem.open(Path(output_path) / "otsu_mask.png", "wb") as f:
        Image.fromarray(otsu_mask).save(f, "png")

    logger.info("Saving stain thumbnail")
    deconv_sample_arr = pull_stain_channel(sample_arr_filtered, vectors=stain_vectors)
    with output_filesystem.open(
        Path(output_path) / "deconv_sample_arr.png",
        "wb",
    ) as f:
        Image.fromarray(deconv_sample_arr).save(f, "png")

    logger.info("Saving stain masks")
    stain0_mask = np.where(deconv_sample_arr[..., 0] > threshold_stain0, 255, 0).astype(
        np.uint8
    )
    stain1_mask = np.where(deconv_sample_arr[..., 1] > threshold_stain1, 255, 0).astype(
        np.uint8
    )
    with output_filesystem.open(
        Path(output_path) / "stain0_mask.png",
        "wb",
    ) as f:
        Image.fromarray(stain0_mask).save(f, "png")
    with output_filesystem.open(
        Path(output_path) / "stain1_mask.png",
        "wb",
    ) as f:
        Image.fromarray(stain1_mask).save(f, "png")

    if filter_query:

        def f_many(iterator, tile_fn, **kwargs):
            return [tile_fn(tile=x, **kwargs) for x in iterator]

        chunks = db.from_sequence(
            tiles_df.itertuples(name="Tile"), partition_size=batch_size
        )
        results = {}
        if "otsu_score" in filter_query:
            logger.info(f"Starting otsu thresholding, threshold={threshold}")

            # chunks = grouper(tiles_df.itertuples(name="Tile"), batch_size)
            results["otsu_score"] = chunks.map_partitions(
                partial(f_many, tile_fn=compute_otsu_score),
                slide_path=slide_urlpath,
                otsu_threshold=threshold,
            )
        if "purple_score" in filter_query:
            logger.info("Starting purple scoring")
            results["purple_score"] = chunks.map_partitions(
                partial(f_many, tile_fn=compute_purple_score), slide_path=slide_urlpath
            )
        if "stain0_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=0, threshold={threshold_stain0}"
            )
            results["stain0_score"] = chunks.map_partitions(
                partial(f_many, tile_fn=compute_stain_score),
                vectors=stain_vectors,
                channel=0,
                stain_threshold=threshold_stain0,
                slide_path=slide_urlpath,
            )
        if "stain1_score" in filter_query:
            logger.info(
                f"Starting stain thresholding, channel=1, threshold={threshold_stain1}"
            )
            results["stain1_score"] = chunks.map_partitions(
                partial(f_many, tile_fn=compute_stain_score),
                vectors=stain_vectors,
                channel=1,
                stain_threshold=threshold_stain1,
                slide_path=slide_urlpath,
            )

        for k, v in results.items():
            tiles_df[k] = v.compute()
        logger.info(f"Filtering based on query: {filter_query}")
        tiles_df = tiles_df.query(filter_query)

    logger.info(tiles_df)

    with output_filesystem.open(tiles_output_path, "wb") as of:
        logger.info(f"saving to {tiles_output_path}")
        tiles_df.to_parquet(of)

    properties = {
        "tiles_url": output_filesystem.unstrip_protocol(tiles_output_path),
        "total_tiles": len(tiles_df),
    }
    return properties


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
