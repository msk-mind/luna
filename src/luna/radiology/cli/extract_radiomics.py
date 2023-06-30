import os
from pathlib import Path
from typing import List

import fire
import medpy.io
import numpy as np
import pandas as pd
from loguru import logger
from radiomics import (
    featureextractor,  # This module is used for interaction with pyradiomics
)


def extract_radiomics_multiple_labels(
    itk_volume_urlpath: str,
    itk_labels_urlpath: str,
    lesion_indices: List[int],
    pyradiomics_config: dict,
    output_urlpath: str,
    check_geometry_strict: bool = False,
    enable_all_filters: bool = False,
):
    """Extract radiomics using pyradiomics, with additional checking of the input geometry match, resulting in a feature csv file

    Args:
        itk_volume_urlpath (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        itk_labels_urlpath (str): path to itk compatible label volume (.mha)
        output_urlpath (str): output/working directory
        lesion_indices (List[int]): list of lesion indicies (label values) to process
        pyradiomics_config (dict): keyword arguments to pass to pyradiomics featureextractor
        check_geometry_strict (str): enforce strict match of the input geometries (otherwise, possible mismatch is silently ignored)
        enable_all_filters (bool): turns on all image filters

    Returns:
        dict: metadata about function call
    """

    if type(lesion_indices) == int:
        lesion_indices = [lesion_indices]

    image, image_header = medpy.io.load(itk_volume_urlpath)

    if Path(itk_labels_urlpath).is_dir():
        label_path_list = [str(path) for path in Path(itk_labels_urlpath).glob("*")]
    elif Path(itk_labels_urlpath).is_file():
        label_path_list = [itk_labels_urlpath]
    else:
        raise RuntimeError("Issue with detecting label format")

    available_labels = set()
    for label_path in label_path_list:
        label, label_header = medpy.io.load(label_path)

        available_labels.update

        available_labels.update(np.unique(label))

        logger.info(f"Checking {label_path}")

        if (
            check_geometry_strict
            and not image_header.get_voxel_spacing() == label_header.get_voxel_spacing()
        ):
            raise RuntimeError(
                f"Voxel spacing mismatch, image.spacing={image_header.get_voxel_spacing()}, label.spacing={label_header.get_voxel_spacing()}"
            )

        if not image.shape == label.shape:
            raise RuntimeError(
                f"Shape mismatch: image.shape={image.shape}, label.shape={label.shape}"
            )

    df_result = pd.DataFrame()

    for lesion_index in available_labels.intersection(lesion_indices):
        extractor = featureextractor.RadiomicsFeatureExtractor(
            label=lesion_index, **pyradiomics_config
        )

        if enable_all_filters:
            extractor.enableAllImageTypes()

        for label_path in label_path_list:
            result = extractor.execute(itk_volume_urlpath, label_path)

            result["lesion_index"] = lesion_index

            df_result = pd.concat([df_result, pd.Series(result).to_frame()], axis=1)

    output_filename = os.path.join(output_urlpath, "radiomics.csv")

    df_result.T.to_csv(output_filename, index=False)

    logger.info(df_result.T)

    properties = {
        "feature_csv": output_filename,
        "lesion_indices": lesion_indices,
    }

    return properties


if __name__ == "__main__":
    fire.Fire(extract_radiomics_multiple_labels)
