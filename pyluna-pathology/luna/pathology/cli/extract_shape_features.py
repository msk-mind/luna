# General imports
import os
import logging
from typing import List
import click
import pandas as pd
import tifffile

from skimage import measure

from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

init_logger()
logger = logging.getLogger("extract_shape_features")

_params_ = [("input_slide_mask", str), ("output_dir", str), ("label_cols", List[str])]


@click.command()
@click.argument("input_slide_mask", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-lc",
    "--label_cols",
    required=False,
    help="columns whose values are used to generate the mask",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
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

    \b
    Inputs:
        input_slide_mask: an input slide mask (*.tif)
    \b
    Outputs:
        shape features for each region
    \b
    Example:
        extract_shape_features ./10001/label_mask.tif
            -lc Background,Tumor
            -o ./shape_features.csv

    """
    cli_runner(cli_kwargs, _params_, extract_shape_features)


def extract_shape_features(
    input_slide_mask: str, label_cols: List[str], output_dir: str
):
    """Extracts shape and spatial features (HIF features) from a slide mask

     Args:
        input_slide_mask (str): path to slide mask (*.tif)
        label_cols (List[str]): list of labels that coorespond to those in input_slide_tiles
        output_dir (str): output/working directory

    Returns:
        dict: output .tif path and the number of shapes for which features were generated 
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

    mask = tifffile.imread(input_slide_mask)
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

    output_fpath = os.path.join(output_dir, "shape_features.csv")
    result_df.to_csv(output_fpath)

    properties = {"shape_features": output_fpath, "num_shapes": len(result_df)}

    logger.info(properties)
    return properties


if __name__ == "__main__":
    cli()
