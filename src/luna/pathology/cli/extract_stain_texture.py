# General imports
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import fire
import fsspec
import numpy as np
import pandas as pd
import scipy.stats
import tiffslide
from fsspec import open
from loguru import logger
from PIL import Image
from tqdm import tqdm

from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.common.utils import (
    extract_patch_texture_features,
    get_downscaled_thumbnail,
    get_full_resolution_generator,
    get_stain_vectors_macenko,
)


@timed
@save_metadata
def cli(
    slide_image_urlpath: str = "???",
    slide_mask_urlpath: str = "???",
    stain_sample_factor: float = "???",  # type: ignore
    stain_channel: int = "???",  # type: ignore
    tile_size: int = "???",  # type: ignore
    output_urlpath: str = ".",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Compute GLCM texture features on a de-convolved slide image

    Args:
        slide_image_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        slide_mask_urlpath (str): url/path to slide mask (.tif)
        stain_sample_factor (float): downsample factor to use for stain vector estimation
        stain_channel (int): which channel of the deconvovled image to use for texture analysis
        tile_size (int): size of tiles to use (at the requested magnification) (500-1000 recommended)
        output_urlpath (str): output/working directory

    Returns:
        dict: metadata about function call

    """
    config = get_config(vars())
    df_result = extract_stain_texture(
        config["slide_image_urlpath"],
        config["slide_mask_urlpath"],
        config["stain_sample_factor"],
        config["stain_channel"],
        config["tile_size"],
        config["output_urlpath"],
        config["storage_options"],
        config["output_storage_options"],
    )

    fs, urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_filename = Path(urlpath_prefix) / "stainomics.parquet"
    with fs.open(output_filename, "wb") as of:
        df_result.to_parquet(of, index=False)

    properties = {
        # "num_pixel_observations": n,
        "feature_data": output_filename,
    }

    return properties


Image.MAX_IMAGE_PIXELS = None


def extract_stain_texture(
    slide_image_urlpath: str,
    slide_mask_urlpath: str,
    stain_sample_factor: float,
    stain_channel: int,
    tile_size: int,
    output_urlpath: str,
    storage_options: dict,
    output_storage_options: dict,
):
    """Compute GLCM texture after automatically deconvolving the image into stain channels, using tile-based processing

    Runs statistics on distribution.

    Save a feature csv file at the output directory.

    Args:
        slide_image_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        slide_mask_urlpath (str): url/path to slide mask (.tif)
        stain_sample_factor (float): downsample factor to use for stain vector estimation
        stain_channel (int): which channel of the deconvovled image to use for texture analysis
        tile_size (int): size of tiles to use (at the requested magnification) (500-1000 recommended)
        output_urlpath (str): output/working URL/path prefix
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions

    Returns:
        dict: metadata about function call
    """
    slide_file = open(slide_image_urlpath, "rb", **storage_options)
    slide = tiffslide.TiffSlide(slide_file)
    # oslide = openslide.OpenSlide(slide_image_urlpath)

    logger.info(f"Slide dimensions {slide.dimensions}")
    sample_arr = get_downscaled_thumbnail(slide, stain_sample_factor)
    slide_full_generator, slide_full_level = get_full_resolution_generator(
        slide_file, tile_size=tile_size
    )

    mask_file = open(slide_mask_urlpath, "rb", **storage_options)
    mask = tiffslide.TiffSlide(mask_file)
    logger.info(f"Mask dimensions {mask.dimensions}")
    mask_full_generator, mask_full_level = get_full_resolution_generator(
        mask_file, tile_size=tile_size
    )

    stain_vectors = get_stain_vectors_macenko(sample_arr)

    logger.info(f"Stain vectors={stain_vectors}")

    tile_x_count, tile_y_count = slide_full_generator.level_tiles[slide_full_level]
    logger.info("Tiles x %s, Tiles y %s", tile_x_count, tile_y_count)

    # populate address, coordinates
    address_raster = [
        address
        for address in itertools.product(range(tile_x_count), range(tile_y_count))
    ]
    logger.info("Number of tiles in raster: %s", len(address_raster))

    features = defaultdict(list)

    N_tiles = len(address_raster)
    for n_tile, address in tqdm(enumerate(address_raster), file=sys.stdout):
        mask_patch = np.array(mask_full_generator.get_tile(mask_full_level, address))

        if not np.count_nonzero(mask_patch) > 1:
            continue

        image_patch = np.array(slide_full_generator.get_tile(slide_full_level, address))

        texture_values = extract_patch_texture_features(
            image_patch,
            mask_patch,
            stain_vectors,
            stain_channel,
            plot=False,
        )

        if texture_values is not None:
            for key, values in texture_values.items():
                features[key].append(values)
        logger.info(f"Processed Tile [{n_tile} / {N_tiles}] at {address}")
    for key, values in features.items():
        features[key] = np.concatenate(values).flatten()
    print(features)

    hist_features = {}
    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        output_urlpath, **output_storage_options
    )
    for key, values in features.items():
        output_path = Path(output_urlpath_prefix) / f"feature_vector_{key}.npy"
        with fs.open(output_path, "wb") as of:
            np.save(of, values)

        if not len(values) > 0:
            continue

        n, (smin, smax), sm, sv, ss, sk = scipy.stats.describe(values)

        if np.min(values) > 0:
            ln_params = scipy.stats.lognorm.fit(values, floc=0)
        else:
            ln_params = (0, 0, 0)

        fx_name_prefix = f"{key}_channel_{stain_channel}"
        hist_features.update(
            {
                f"{fx_name_prefix}_nobs": n,
                f"{fx_name_prefix}_min": smin,
                f"{fx_name_prefix}_max": smax,
                f"{fx_name_prefix}_mean": sm,
                f"{fx_name_prefix}_variance": sv,
                f"{fx_name_prefix}_skewness": ss,
                f"{fx_name_prefix}_kurtosis": sk,
                f"{fx_name_prefix}_lognorm_fit_p0": ln_params[0],
                f"{fx_name_prefix}_lognorm_fit_p2": ln_params[2],
            }
        )

    # The fit may fail sometimes, replace inf with 0
    df_result = (
        pd.DataFrame(data=hist_features, index=[0])
        .replace([np.inf, -np.inf], 0.0)
        .astype(float)
    )
    logger.info(df_result)
    slide_file.close()
    mask_file.close()

    return df_result


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
