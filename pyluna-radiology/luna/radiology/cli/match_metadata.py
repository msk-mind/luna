import logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('match_metadata')

from luna.common.utils import cli_runner

from typing import List
_params_ = [('input_image_data', str), ('input_label_data', str), ('output_dir', str)]

@click.command()
@click.option('-ii', '--input_image_data', required=False,
              help='path to input image data')
@click.option('-il', '--input_label_data', required=False,
              help='path to input label data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Resamples and co-registeres two input volumes to occupy the same physical coordinates

    \b
        coregister_volumes
            --input_image_data volume_ct.nii
            --input_label_data volume_pet.nii
            -o ./registered/
    """
    cli_runner(cli_kwargs, _params_, match_metadata )

import medpy.io
from pydicom import dcmread

def match_metadata(dicom_tree_folder, input_label_data):
    dicom_folders = set()
    for dicom in Path(dicom_tree_folder).rglob("*.dcm"):
        dicom_folders.add(dicom.parent)
    
    label, label_header = medpy.io.load(input_label_data)
    found_dicom_path = None
    for dicom_folder in dicom_folders:
        n_slices_dcm = len(list((dicom_folder).glob("*.dcm")))
        if label.shape[2] == n_slices_dcm: found_dicom_path = dicom_folder

    if found_dicom_path: 
        path = next(found_dicom_path.glob("*.dcm"))
        ds = dcmread(path)
        print ("Matched: z=", label.shape[2], found_dicom_path, ds.PatientName, ds.AccessionNumber, ds.StudyDescription, ds.SeriesDescription, ds.SliceThickness)
    return str(found_dicom_path)

if __name__ == "__main__":
    cli()
