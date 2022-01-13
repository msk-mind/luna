import logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('match_metadata')

from luna.common.utils import cli_runner

from typing import List
_params_ = [('dicom_tree_folder', str), ('input_itk_labels', str), ('output_dir', str)]

@click.command()
@click.argument('dicom_tree_folder', nargs=1)
@click.argument('input_itk_labels', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Scans a dicom tree for images that match the z-dimension (number of slices), useful to matching case segmentations to their associated scans

    \b
    Input:
        dicom_tree_folder: path a dicom tree folder containing many scans and many dicoms
    \b
    Output:
        dicom_folder: path to a single dicom series
    \b
    Example:
        match_metadata ./scan_folder/ scan_segmentation.nii
            -o ./matched_dicoms/
    """
    cli_runner(cli_kwargs, _params_, match_metadata )

import medpy.io
from pydicom import dcmread
from pathlib import Path

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
        if label.shape[2] == n_slices_dcm: found_dicom_paths.add (dicom_folder)

    # logger.info(found_dicom_paths)
    # if not len(found_dicom_paths)==1:
    #     raise RuntimeError("Could not find unique matching scans!")

    for found_dicom_path in found_dicom_paths: 
        path = next(found_dicom_path.glob("*.dcm"))
        ds = dcmread(path)
        print ("Matched: z=", label.shape[2], found_dicom_path, ds.PatientName, ds.AccessionNumber, ds.StudyDescription, ds.SeriesDescription, ds.SliceThickness)
    
    properties = {
        'dicom_folder': str(found_dicom_path),
        'zdim': label.shape[2],
    }

    return properties

if __name__ == "__main__":
    cli()
