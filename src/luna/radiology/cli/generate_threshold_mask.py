
import fire
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from loguru import logger

from luna.radiology.mirp.imageReaders import read_itk_image


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
    fire.Fire(generate_threshold_mask)
