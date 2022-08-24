import os, logging
import click
from pydicom import dcmread

from luna.common.custom_logger   import init_logger, add_log_dir

init_logger()
logger = logging.getLogger('extract_dicom_headers')

from luna.common.utils import cli_runner

_params_ = [('input_dicom_folder', str), ('output_dir', str)]

@click.command()
@click.argument('input_dicom_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-it', '--itk_image_type', required=False,
              help="desired ITK image extention")
@click.option('-ct', '--itk_c_type', required=False,
              help="desired C datatype (float, unsigned short)")   
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Generates a ITK compatible image from a dicom series
    
    \b
    Inputs:
        input_dicom_folder: A folder containing a dicom series
    \b
    Outputs:
        itk_volume
    \b
    Example:
        dicom_to_itk ./scans/10000/2/DICOM/
            --itk_image_type nrrd
            --itk_c_type 'unsigned short'
            -o ./scans/10000/2/NRRD
    """
    cli_runner(cli_kwargs, _params_, extract_dicom_headers)

from pathlib import Path
import pydicom
from pydicom import dcmread

import pandas as pd

def parse_dicom(dcm_path):
    ds = dcmread(dcm_path)
    kv = {}
    types = set()
    skipped_keys = []

    for elem in ds.iterall():
        types.add(type(elem.value))
        if type(elem.value) in [int, float, str]:
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal, pydicom.valuerep.IS, pydicom.valuerep.PersonName, pydicom.uid.UID]:
            kv[elem.keyword] = str(elem.value)
        elif type(elem.value) in [list, pydicom.multival.MultiValue]:
            kv[elem.keyword] = "//".join([str(x) for x in elem.value])
        else:
            skipped_keys.append(elem.keyword)

    if "" in kv:
        kv.pop("")
    return kv

def extract_dicom_headers(input_dicom_folder, output_dir):
    """Generate an ITK compatible image from a dicom series/folder

    Args:
        input_dicom_folder (str): path to dicom series within a folder
        output_dir (str): output/working directory
        itk_image_type (str): ITK volume image type to output (mhd, nrrd, nii, etc.)
        itk_c_type (str): pixel (C) type for pixels, e.g. float or unsigned short

    Returns:
        dict: metadata about function call
    """
    add_log_dir(logger, output_dir)

    dcms = list(Path(input_dicom_folder).glob("*.dcm"))

    dicom_kvs = []

    for dcm in dcms: dicom_kvs.append ( parse_dicom(dcm) )

    output_ds = f"{output_dir}/scan_metadata.parquet"

    df = pd.DataFrame(dicom_kvs)

    logger.info(df)
    
    df.to_parquet(output_ds, index=False)
    
    properties = {
        "feature_data": output_ds,
        "segment_keys": {
            "radiology_patient_name": df.loc[0, "PatientName"],
            "radiology_accession_number": df.loc[0, "AccessionNumber"],
            "radiology_series_instance_uuid": df.loc[0, "SeriesInstanceUID"],
        }
    }

    return properties


if __name__ == "__main__":
    cli()

