# General imports
from pathlib import Path
from typing import Dict, List

import fire
import fsspec
import numpy as np
import pandas as pd
import tifffile
import tiffslide
from fsspec import open
from loguru import logger
from pandera.typing import DataFrame
from skimage import measure
from scipy.stats import entropy

from luna.common.models import LabeledTileSchema
from luna.common.utils import get_config, save_metadata, timed
from luna.pathology.cli.generate_tile_mask import convert_tiles_to_mask


@timed
@save_metadata
def cli(
    slide_mask_urlpath: str = "???",
    label_cols: List[str] = "???",  # type: ignore
    output_urlpath: str = "???",  # type: ignore
    include_smaller_regions: bool = False,
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

    with open(config["slide_mask_urlpath"], "rb", **config["storage_options"]) as of:
        mask = tifffile.imread(of)

    mask_values = {k: v + 1 for v, k in enumerate(config["label_cols"])}
    result_df = extract_shape_features(mask, mask_values, config['include_smaller_regions'])

    fs, urlpath = fsspec.core.url_to_fs(
        config["output_urlpath"], **config["output_storage_options"]
    )

    output_fpath = Path(urlpath) / "shape_features.csv"
    with fs.open(output_fpath, "w") as of:
        result_df.to_csv(of)

    properties = {"shape_features": output_fpath, "num_shapes": len(result_df)}

    logger.info(properties)
    return properties


def extract_whole_slide_features(
    mask: np.ndarray,
    mask_values: Dict[int, str],
    properties: List[str] = [
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
    ],
):
    logger.info(f"Mapping mask values to labels: {mask_values}")
    label_mapper = {v: k for k, v in mask_values.items()}

    value, counts = np.unique(mask, return_counts=True)


    logger.info("Extracting whole slide features")
    # gathering whole slide features, one vector per label
    whole_slide_features = measure.regionprops_table(
        label_image=mask, properties=properties
    )
    whole_slide_features_df = pd.DataFrame.from_dict(whole_slide_features)

    if 'perimeter' in whole_slide_features_df.columns and 'area' in whole_slide_features_df.columns:
        whole_slide_features_df['perimeter_area_ratio'] = whole_slide_features_df['perimeter'] / whole_slide_features_df['area']

    # add column with label name
    whole_slide_features_df["Class"] = whole_slide_features_df["label"].map(
        label_mapper
    )
    whole_slide_features_df = whole_slide_features_df.drop('label', axis=1)
    logger.info(
        f"Extracted whole slide features for {len(whole_slide_features_df)} labels"
    )

    return whole_slide_features_df


def extract_regional_features(
    mask: np.ndarray,
    mask_values: Dict[int, str],
    properties: List[str] = [
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
        "min_intensity",
        "max_intensity",
    ],
):
    label_mapper = {v: k for k, v in mask_values.items()}

    mask_label = measure.label(mask, connectivity=2)
    regional_features = measure.regionprops_table(
        label_image=mask_label, intensity_image=mask, properties=properties
    )
    regional_features_df = pd.DataFrame.from_dict(regional_features)

    if 'perimeter' in regional_features_df.columns and 'area' in regional_features_df.columns:
        regional_features_df['perimeter_area_ratio'] = regional_features_df['perimeter'] / regional_features_df['area']

    # add column with label name
    regional_features_df["Class"] = regional_features_df["min_intensity"].map(
        label_mapper
    )
    regional_features_df = regional_features_df.drop(
        columns=["max_intensity", "min_intensity"]
    )
    regional_features_df = regional_features_df.drop('label', axis=1)

    logger.info(f"Extracted regional features for {len(regional_features_df)} regions")

    return regional_features_df


def extract_shape_features(
    mask: np.ndarray,
    mask_values: Dict[int, str],
    include_smaller_regions = False,
    properties: List[str] = [
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
    ],
):
    """Extracts shape and spatial features (HIF features) from a slide mask

     Args:
        slide_mask_urlpath (str): url/path to slide mask (*.tif)
        label_cols (List[str]): list of labels that coorespond to those in slide_mask_urlpath

    Returns:
        pd.DataFrame: shape and spatial features
    """

    logger.info(f"Mask shape={mask.shape}")

    logger.info("Extracting regional features based on connectivity")
    whole_slide_features_df = extract_whole_slide_features(mask, mask_values, properties)
    whole_slide_features_df['Parent'] = 'whole_region'
    whole_slide_features_df = whole_slide_features_df.set_index('Class')
    whole_slide_features_df['area_fraction'] = whole_slide_features_df['area'] / whole_slide_features_df['area'].sum()
    whole_slide_features_mdf = pd.melt(whole_slide_features_df.reset_index(), id_vars=['Parent', 'Class'])

    area_col = whole_slide_features_df.columns.get_loc("area")
    idx0, idx1 = np.triu_indices(len(whole_slide_features_df), 1)
    np.seterr(divide="ignore")
    whole_slide_ratio_df = pd.DataFrame(
        data={
            "Parent": "whole_region",
            "variable": np.array(
                [
                    f"area_log_ratio_to_{row}" for row in whole_slide_features_df.index.values
                ]
            )[idx1],
            "value": np.log(whole_slide_features_df.iloc[idx0, area_col].values)
            - np.log(whole_slide_features_df.iloc[idx1, area_col].values),
        },
        index=whole_slide_features_df.index[idx0],
    )
    whole_slide_ratio_df = whole_slide_ratio_df.reset_index()

    properties += ['min_intensity', 'max_intensity']
    regional_features_df = extract_regional_features(mask, mask_values, properties)
    regional_features_df = regional_features_df.assign(Parent=[f'region_{x}' for x in range(len(regional_features_df))])
    regional_features_df = regional_features_df.set_index(['Parent', 'Class'])
    regional_features_df['area_fraction'] = regional_features_df['area'] / whole_slide_features_df['area']
    regional_features_mdf = pd.melt(regional_features_df.reset_index(), id_vars=['Parent', 'Class'])

    regional_features_df = regional_features_df.reset_index()
    largest_regional_features_df = regional_features_df.loc[regional_features_df.groupby('Class')['area'].idxmax()]
    largest_regional_features_df['Parent'] = 'largest_region'
    largest_regional_features_df = largest_regional_features_df.set_index('Class')
    largest_regional_features_mdf = pd.melt(largest_regional_features_df.reset_index(), id_vars=['Parent', 'Class'])

    area_col = largest_regional_features_df.columns.get_loc("area")
    idx0, idx1 = np.triu_indices(len(largest_regional_features_df), 1)
    np.seterr(divide="ignore")
    ratio_df = pd.DataFrame(
        data={
            "Parent": "largest_region",
            "variable": np.array(
                [
                    f"area_log_ratio_to_{row}" for row in largest_regional_features_df.index.values
                ]
            )[idx1],
            "value": np.log(largest_regional_features_df.iloc[idx0, area_col].values)
            - np.log(largest_regional_features_df.iloc[idx1, area_col].values),
        },
        index=largest_regional_features_df.index[idx0],
    )
    ratio_df = ratio_df.reset_index()

    result_df = pd.concat([whole_slide_features_mdf, whole_slide_ratio_df,
                           largest_regional_features_mdf, ratio_df])

    if include_smaller_regions:
        result_df = pd.concat([result_df, regional_features_mdf])

    return result_df


def fire_cli():
    fire.Fire(cli)


if __name__ == "__main__":
    fire_cli()
