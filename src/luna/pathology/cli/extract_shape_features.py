# General imports
import logging
from pathlib import Path
from typing import List

import fire
import fsspec
import pandas as pd
import tifffile
from fsspec import open
from skimage import measure

from luna.common.custom_logger import init_logger
from luna.common.utils import get_config, save_metadata, timed

init_logger()
logger = logging.getLogger("extract_shape_features")


@timed
@save_metadata
def cli(
    slide_mask_urlpath: str = "???",
    label_cols: List[str] = "???",  # type: ignore
    output_urlpath: str = "???",  # type: ignore
    storage_options: dict = {},
    output_storage_options: dict = {},
    local_config: str = "",
):
    """Extracts shape and spatial features (HIF features) from a slide mask.
    This CLI extracts two sets of features. The first set are 'whole slide features', where
    the entire mask label is considred as a single region and features are extracted. These features
    are useful for determining things like total area of x tissue.

    The second set of features are 'regional features', where each label is split up according to
    their connectivity and features are extracted from these smaller regions.
    These features are useful for determining things like solidity of the top ten largest
    regions of tissue y. Pixel intensity values from the WSI are unused. In order to generate
    connected regions, skimage generates a mask itself where different values coorespond
    to different regions, which removes the tissue type information from the original mask.
    So, the original mask is passed as an intensity image to ensure that each region can be
    associated with a tissue type.

     Args:
        slide_mask_urlpath (str): URL/path to slide mask (*.tif)
        label_cols (List[str]): list of labels that coorespond to those in slide_mask_urlpath
        output_urlpath (str): output URL/path prefix

    Returns:
        dict: output .tif path and the number of shapes for which features were generated

    """
    config = get_config(vars())
    result_df = extract_shape_features(
        config["slide_mask_urlpath"], config["label_cols"], config["storage_options"]
    )

    fs, urlpath = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    output_fpath = Path(urlpath) / "shape_features.csv"
    with fs.open(output_fpath, "w") as of:
        result_df.to_csv(of)

    properties = {"shape_features": output_fpath, "num_shapes": len(result_df)}

    logger.info(properties)
    return properties


def extract_shape_features(
    slide_mask_urlpath: str,
    label_cols: List[str],
    storage_options: dict,
):
    """Extracts shape and spatial features (HIF features) from a slide mask

     Args:
        slide_mask_urlpath (str): url/path to slide mask (*.tif)
        label_cols (List[str]): list of labels that coorespond to those in slide_mask_urlpath

    Returns:
        pd.DataFrame: shape and spatial features
    """
    # List of features to extract.
    # Default behavior of regionprops_table only generates label and bbox features.
    # Not all of these features may be relevant
    properties = [
        "area",
        "bbox",
        "bbox_area",
        "centroid",
        "convex_area",
        "convex_image",
        "coords",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "filled_area",
        "filled_image",
        "image",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "label",
        "local_centroid",
        "major_axis_length",
        "minor_axis_length",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "orientation",
        "perimeter",
        "slice",
        "solidity",
    ]

    with open(slide_mask_urlpath, "rb", **storage_options).open() as of:
        mask = tifffile.imread(of)
    logger.info(f"Mask shape={mask.shape}")

    mask_values = {k: v + 1 for v, k in enumerate(label_cols)}
    logger.info(f"Mapping mask values to labels: {mask_values}")
    label_mapper = {v: k for k, v in mask_values.items()}

    logger.info("Extracting whole slide features")
    # gathering whole slide features, one vector per label
    whole_slide_features = measure.regionprops_table(
        label_image=mask, properties=properties
    )
    whole_slide_features_df = pd.DataFrame.from_dict(whole_slide_features)

    # add column with label name
    whole_slide_features_df["label_name"] = "whole_" + whole_slide_features_df[
        "label"
    ].map(label_mapper)
    logger.info(
        f"Extracted whole slide features for {len(whole_slide_features_df)} labels"
    )

    logger.info("Extracting regional features based on connectivity")
    mask_label = measure.label(mask, connectivity=2)
    # pass intensity image to propogate label type
    properties.extend(
        [
            "min_intensity",
            "max_intensity",
        ]
    )
    regional_features = measure.regionprops_table(
        label_image=mask_label, intensity_image=mask, properties=properties
    )
    regional_features_df = pd.DataFrame.from_dict(regional_features)

    # add column with label name
    regional_features_df["label_name"] = regional_features_df["min_intensity"].map(
        label_mapper
    )
    regional_features_df = regional_features_df.drop(
        columns=["max_intensity", "min_intensity"]
    )

    logger.info(f"Extracted regional features for {len(regional_features_df)} regions")

    result_df = pd.concat([whole_slide_features_df, regional_features_df])

    return result_df


if __name__ == "__main__":
    fire.Fire(cli)
