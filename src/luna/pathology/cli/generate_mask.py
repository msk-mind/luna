# General imports
from pathlib import Path

import fire
import fsspec
import numpy as np
import openslide
import pandas as pd
import tifffile
import tiffslide
from fsspec import open
from loguru import logger
from PIL import Image
from skimage.measure import block_reduce

from luna.common.utils import get_config, local_cache_urlpath, save_metadata, timed
from luna.pathology.common.utils import convert_xml_to_mask, get_layer_names


@timed
@save_metadata
def cli(
    slide_urlpath: str = "???",
    roi_urlpath: str = "???",
    output_urlpath: str = "???",
    annotation_name: str = "???",
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Generate a full resolution mask image (.tif) from vector annotations (polygons, shapes)

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_roi: roi containing vector shapes (*.annotations, *.json)
    \b
    Outputs:
        slide_mask
    \b
    Example:
        generate_mask ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    config = get_config(vars())
    df = generate_mask(
        config["slide_urlpath"],
        config["roi_urlpath"],
        config["output_urlpath"],
        config["annotation_name"],
        config["storage_options"],
        config["output_storage_options"],
    )

    fs, output_urlpath_prefix = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )
    output_filename = Path(output_urlpath_prefix) / "mask_data.parquet"
    with fs.open(output_filename, "wb") as of:
        df.to_parquet(of)

    slide_id = Path(config["roi_urlpath"]).stem
    properties = {
        "slide_mask": Path(output_urlpath_prefix) / "mask_full_res.tif",
        "feature_data": output_filename,
        "mask_size": df["mask_size"].tolist(),
        "segment_keys": {"slide_id": slide_id},
    }

    return properties


@local_cache_urlpath(
    dir_key_write_mode={
        "output_urlpath": "w",
    }
)
def generate_mask(
    slide_urlpath: str,
    roi_urlpath: str,
    output_urlpath: str,
    annotation_name: str,
    storage_options: dict,
    output_storage_options: dict,
):
    """Generate a full resolution mask image (.tif) from vector annotations (polygons, shapes)

    Take into account positive and negative spaces.  Essentially rasterizes a polygon file.

    Args:
        slide_urlpath (str): slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...) absolute or relative path. prefix with scheme to use alternative file systems.
        roi_urlpath (str):  halo or other polygonal annotation file (.xml, .geojson) absolute or relative path. prefix with scheme to use alternative file systems.
        output_urlpath (str): output/working absolute or relative path. prefix with scheme to use alternative file systems.
        annotation_name (str): name of annotation layer to use
        storage_options (dict): storage options that make sense for the file storage used

    Returns:
        DataFrame: mask properties
    """
    mask_properties = {}

    with open(slide_urlpath, **storage_options) as of:
        slide = tiffslide.TiffSlide(of)
        thumbnail = slide.get_thumbnail((1000, 1000))

    with open(Path(output_urlpath) / "slide_thumbnail.png", "wb") as of:
        thumbnail.save(of, format="PNG")

    wsi_shape = (
        slide.dimensions[1],
        slide.dimensions[0],
    )  # Annotation file has flipped dimensions w.r.t openslide conventions
    logger.info(f"Slide shape={wsi_shape}")

    layer_names = get_layer_names(roi_urlpath, storage_options)
    logger.info(f"Available layer names={layer_names}")

    mask_properties["layer_names"] = list(layer_names)
    mask_properties["mask_size"] = list(wsi_shape)

    mask_arr, xml_region_properties = convert_xml_to_mask(
        roi_urlpath, wsi_shape, annotation_name, storage_options=storage_options
    )

    mask_properties.update(xml_region_properties)

    logger.info(f"Generating mask thumbnail, mask size={mask_arr.shape}")
    mask_thumbnail = openslide.ImageSlide(
        Image.fromarray(
            255 * block_reduce(mask_arr, block_size=(10, 10), func=np.mean, cval=0.0)
        )
    ).get_thumbnail((1000, 1000))

    with open(Path(output_urlpath) / "mask_thumbnail.png", "wb") as of:
        mask_thumbnail.save(of, format="PNG")

    slide_mask_file = Path(output_urlpath) / "mask_full_res.tif"
    with open(slide_mask_file, "wb") as of:
        tifffile.imwrite(of, mask_arr)

    return pd.DataFrame(mask_properties)


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
