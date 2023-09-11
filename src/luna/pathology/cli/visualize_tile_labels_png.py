# General imports
from pathlib import Path
from typing import List, Optional

import fire
import fsspec
import pandas as pd
import tiffslide
from fsspec import open
from loguru import logger
from PIL import Image

from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.cli.generate_tiles import generate_tiles
from luna.pathology.common.utils import (
    get_downscaled_thumbnail,
    get_scale_factor_at_magnification,
    visualize_tiling_scores,
)


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    tiles_urlpath: str = "",
    mpp_units: bool = False,
    plot_labels: List[str] = "???",  # type: ignore
    output_urlpath: str = ".",
    requested_magnification: Optional[int] = None,
    tile_size: Optional[int] = None,
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Generate nice tile markup images with continuous or discrete tile scores

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): url/path to a slide-tile manifest file (.tiles.csv)
        mpp_units (bool): if true, additional rescaling is applied to match micro-meter and pixel coordinate systems
        plot_labels (List[str]): labels to plot
        output_urlpath (str): output url/path prefix
        requested_magnification (int): Magnification scale at which to perform computation
        tile_size (int): tile size
        storage_options (dict): storage options to pass to reading functions
        output_storage_options (dict): storage options to pass to writing functions
        local_config (str): url/path to local config YAML file

    Returns:
        dict: metadata about function call
    """
    config = get_config(vars())

    if not config["tile_size"] and not config["tiles_urlpath"]:
        raise fire.core.FireError("Specify either tiles_urlpath or tile_size")

    thumbnails_overlayed = visualize_tiles(
        config["slide_urlpath"],
        config["tiles_urlpath"],
        config["mpp_units"],
        config["plot_labels"],
        config["requested_magnification"],
        config["tile_size"],
        config["storage_options"],
    )

    fs, output_path_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    images = {}
    for score_type, thumbnail_overlayed in thumbnails_overlayed.items():
        output_file = (
            Path(output_path_prefix)
            / f"tile_scores_and_labels_visualization_{score_type}.png"
        )
        thumbnail_overlayed = Image.fromarray(thumbnail_overlayed)
        with fs.open(output_file, "wb") as of:
            thumbnail_overlayed.save(of, format="PNG")
        images[score_type] = str(output_file)
        logger.info(f"Saved {score_type} visualization at {output_file}")

    properties = {
        "data": fs.unstrip_protocol(output_path_prefix),
        "images": images,
    }

    return properties


def visualize_tiles(
    slide_urlpath: str,
    tiles_urlpath: str,
    mpp_units: bool,
    plot_labels: List[str],
    requested_magnification: Optional[int] = None,
    tile_size: Optional[int] = None,
    storage_options: dict = {},
):
    """Generate nice tile markup images with continuous or discrete tile scores

    Args:
        slide_urlpath (str): url/path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        tiles_urlpath (str): url/path to a slide-tile manifest file (.tiles.csv)
        mpp_units (bool): if true, additional rescaling is applied to match micro-meter and pixel coordinate systems
        plot_labels (List[str]): labels to plot
        requested_magnification (int): Magnification scale at which to perform computation
        tile_size (int): tile size
        storage_options (dict): storage options to pass to reading functions

    Returns:
        dict[str,np.ndarray]: score type to numpy array representation of overlayed thumbnail
    """
    if type(plot_labels) == str:
        plot_labels = [plot_labels]

    # Get tiles
    if tiles_urlpath:
        with open(tiles_urlpath, **storage_options) as of:
            df = pd.read_parquet(of).reset_index().set_index("address")
    elif type(tile_size) == int:
        df = generate_tiles(
            slide_urlpath, tile_size, storage_options, requested_magnification
        )
    else:
        raise RuntimeError("Specify tile size or url/path to tiling data")

    with open(slide_urlpath, **storage_options) as of:
        slide = tiffslide.TiffSlide(of)

        to_mag_scale_factor = get_scale_factor_at_magnification(
            slide, requested_magnification=requested_magnification
        )

        # Create thumbnail image for scoring
        sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)

        # See if we need to adjust scale_factor to account for different units
        if mpp_units:
            unit_sf = 0.0
            for mpp_key in ("aperio.MPP", "openslide.mpp-x"):
                if mpp_key in slide.properties:
                    unit_sf = float(slide.properties[mpp_key])
            if unit_sf:
                to_mag_scale_factor *= unit_sf
            else:
                logger.warning(
                    "No MPP scale factor was recognized in slide properties."
                )

    # only visualize tile scores that were able to be computed
    all_score_types = set(plot_labels)
    score_types_to_visualize = set(list(df.columns)).intersection(all_score_types)

    thumbnails_overlayed = {}  # type: Dict[str,np.ndarray]
    for score_type in score_types_to_visualize:
        thumbnails_overlayed[score_type] = visualize_tiling_scores(
            df, sample_arr, to_mag_scale_factor, score_type
        )

    return thumbnails_overlayed


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
