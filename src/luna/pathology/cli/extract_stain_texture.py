# General imports
import itertools
import os
import sys
from collections import defaultdict

import click
import numpy as np
import openslide
import pandas as pd
import scipy.stats
from PIL import Image
from tqdm import tqdm

from loguru import logger
from luna.common.utils import cli_runner
from luna.pathology.common.utils import (
    extract_patch_texture_features,
    get_downscaled_thumbnail,
    get_full_resolution_generator,
    get_stain_vectors_macenko,
)



_params_ = [
    ("input_slide_image", str),
    ("input_slide_mask", str),
    ("output_dir", str),
    ("stain_sample_factor", int),
    ("tile_size", int),
    ("stain_channel", int),
]


@click.command()
@click.argument("input_slide_image", nargs=1)
@click.argument("input_slide_mask", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-tx",
    "--tile_size",
    required=False,
    help="size of tiles to use as inputs to the glcm calculation",
)
@click.option(
    "-sc",
    "--stain_channel",
    required=False,
    help="which stain channel to use for texture features",
)
@click.option(
    "-sf",
    "--stain_sample_factor",
    required=False,
    help="downsample factor for the image used in stain vector estimation",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """Compute GLCM texture features on a de-convolved slide image

    \b
    Inputs:
        input_slide_image: slide image (.svs)
        input_slide_mask: whole-slide mask image (.tif)
    \b
    Outputs:
        feature_csv
    \b
    Example:
        extract_stain_texture ./slides/10001.svs ./masks/10001/tumor_mask.tif
            -tx 500 -sc 0 -sf 10 -o ./stain_features/10001/
    """
    cli_runner(cli_kwargs, _params_, extract_stain_texture)


Image.MAX_IMAGE_PIXELS = None


def extract_stain_texture(
    input_slide_image,
    input_slide_mask,
    stain_sample_factor,
    stain_channel,
    tile_size,
    output_dir,
):
    """Compute GLCM texture after automatically deconvolving the image into stain channels, using tile-based processing

    Runs statistics on distribution.

    Save a feature csv file at the output directory.

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        output_dir (str): output/working directory
        stain_sample_factor (float): downsample factor to use for stain vector estimation
        stain_channel (int): which channel of the deconvovled image to use for texture analysis
        tile_size (int): size of tiles to use (at the requested magnification) (500-1000 recommended)

    Returns:
        dict: metadata about function call
    """
    slide = openslide.OpenSlide(input_slide_image)
    mask = openslide.ImageSlide(input_slide_mask)

    logger.info(
        f"Slide dimensions {slide.dimensions}, Mask dimensions {mask.dimensions}"
    )

    sample_arr = get_downscaled_thumbnail(slide, stain_sample_factor)
    stain_vectors = get_stain_vectors_macenko(sample_arr)

    logger.info(f"Stain vectors={stain_vectors}")

    slide_full_generator, slide_full_level = get_full_resolution_generator(
        slide, tile_size=tile_size
    )
    mask_full_generator, mask_full_level = get_full_resolution_generator(
        mask, tile_size=tile_size
    )

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
        mask_patch = np.array(mask_full_generator.get_tile(mask_full_level, address))[
            :, :, 0
        ]

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
    for key, values in features.items():
        np.save(os.path.join(output_dir, f"feature_vector_{key}.npy"), values)

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

    output_filename = os.path.join(output_dir, "stainomics.parquet")

    df_result.to_parquet(output_filename, index=False)

    properties = {
        "num_pixel_observations": n,
        "feature_data": output_filename,
    }

    return properties


if __name__ == "__main__":
    cli()