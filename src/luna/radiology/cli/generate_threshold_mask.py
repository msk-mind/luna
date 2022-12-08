import logging
import os

import click

from luna.common.custom_logger import init_logger

init_logger()
logger = logging.getLogger("generate_threshold_mask")

from luna.common.utils import cli_runner

_params_ = [
    ("input_itk_volume", str),
    ("threshold", float),
    ("area_closing_radius", float),
    ("expansion_radius", float),
    ("output_dir", str),
    ("save_npy", bool),
]


@click.command()
@click.argument("input_itk_volume", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-th",
    "--threshold",
    required=False,
    help="Threshold above which voxels are labeled",
)
@click.option(
    "-ar",
    "--area_closing_radius",
    required=False,
    help="Radius (of a sphere) of areas to close",
)
@click.option(
    "-er",
    "--expansion_radius",
    required=False,
    help="Radius (in mm) to grow segmentation mask",
)
@click.option(
    "-npy",
    "--save_npy",
    required=False,
    is_flag=True,
    help="whether to additionally save the volumes as numpy files",
)
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
@click.option(
    "-dsid",
    "--dataset_id",
    required=False,
    help="Optional dataset identifier to add results to",
)
def cli(**cli_kwargs):
    """ """
    cli_runner(cli_kwargs, _params_, generate_threshold_mask)


import numpy as np
import pandas as pd
from scipy.ndimage import (binary_dilation, distance_transform_edt,
                           generate_binary_structure)
from skimage.morphology import area_closing, binary_erosion, remove_small_holes

from luna.radiology.mirp.imageReaders import (read_itk_image,
                                              read_itk_segmentation)


def generate_threshold_mask(
    input_itk_volume,
    threshold,
    area_closing_radius,
    expansion_radius,
    save_npy,
    output_dir,
):
    """
    Use: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.expand_labels
    And: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    """
    d_properties = {}

    image_class_object_volume = read_itk_image(input_itk_volume)

    voxel_size = image_class_object_volume.spacing[0]
    voxel_grid = image_class_object_volume.get_voxel_grid()

    voxel_grid = np.where(voxel_grid > threshold, 1, 0).astype(bool)
    logger.info(f"Initial mask sum = {voxel_grid.sum()}")

    logger.info("Applying area closing....")
    closing_surface_area = 12 * area_closing_radius**2 / voxel_size
    voxel_grid = remove_small_holes(voxel_grid, closing_surface_area)
    logger.info(
        f"area_closing(SA={closing_surface_area}) mask sum = {voxel_grid.sum()}"
    )

    voxel_edt = distance_transform_edt(-voxel_grid.astype(int) + 1, sampling=voxel_size)
    voxel_mask = np.where(voxel_edt < expansion_radius, 1, 0)

    image_class_object_volume.set_voxel_grid(voxel_grid=voxel_edt)

    image_file = image_class_object_volume.export(file_path=output_dir)
    d_properties["itk_labels"] = image_file

    if save_npy:
        np.save(image_file + ".npy", voxel_mask.astype(np.uint8))
        d_properties["npy_labels"] = image_file + ".npy"
        np.save(image_file + ".edt.npy", voxel_edt.astype(np.float32))
        d_properties["npy_edt_labels"] = image_file + ".edt.npy"

    d_properties["n_mask_voxels"] = voxel_grid.sum()

    output_file = f"{output_dir}/volume_registered_features.parquet"
    pd.DataFrame([d_properties]).to_parquet(output_file)

    d_properties["segment_keys"] = {"radiology_submodality": "DERIVED"}

    d_properties["feature_data"] = output_file

    return d_properties


if __name__ == "__main__":
    cli()
