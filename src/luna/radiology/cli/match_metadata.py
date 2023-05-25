import logging
from pathlib import Path

import fire
import medpy.io
from pydicom import dcmread

from luna.common.custom_logger import init_logger

init_logger()
logger = logging.getLogger("match_metadata")


def match_metadata(dicom_tree_folder: str, input_itk_labels: str, output_dir: str):
    """Generate an ITK compatible image from a dicom series/folder

    Args:
        dicom_tree_folder (str): path to root/tree dicom folder that may contained multiple dicom series
        input_itk_labels (str): path to itk compatible label volume (.mha)
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    dicom_folders = set()
    for dicom in Path(dicom_tree_folder).rglob("*.dcm"):
        dicom_folders.add(dicom.parent)

    label, _ = medpy.io.load(input_itk_labels)
    found_dicom_paths = set()
    for dicom_folder in dicom_folders:
        n_slices_dcm = len(list((dicom_folder).glob("*.dcm")))
        if label.shape[2] == n_slices_dcm:
            found_dicom_paths.add(dicom_folder)

    logger.info(found_dicom_paths)
    if not len(found_dicom_paths) > 0:
        raise RuntimeError("Could not find matching scans!")

    matched = None
    for found_dicom_path in found_dicom_paths:
        path = next(found_dicom_path.glob("*.dcm"))
        ds = dcmread(path)
        if not ds.Modality == "CT":
            continue
        print(
            "Matched: z=",
            label.shape[2],
            found_dicom_path,
            ds.PatientName,
            ds.AccessionNumber,
            ds.SeriesInstanceUID,
            ds.StudyDescription,
            ds.SeriesDescription,
            ds.SliceThickness,
        )
        matched = found_dicom_path

    if matched is None:
        raise RuntimeError("Could not find matching CT!")

    properties = {
        "dicom_folder": str(matched),
        "zdim": label.shape[2],
    }

    return properties


if __name__ == "__main__":
    fire.Fire(match_metadata)
