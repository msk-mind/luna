import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('randomize_contours')

from luna.common.utils import cli_runner

_params_ = [('input_itk_volume', str), ('input_itk_labels', str), ('resample_pixel_spacing', float), ('output_dir', str)]

@click.command()
@click.argument('input_itk_volume', nargs=1)
@click.argument('input_itk_labels', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rps', '--resample_pixel_spacing', required=False,
              help='isotropic voxel size (in mm)')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
@click.option('-sk', '--super_key', required=False,
              help='Extra key')
@click.option("-dsid", "--dataset_id", required=False, help="Optional dataset identifier to add tabular output to" )
def cli(**cli_kwargs):
    """Extract ROI
    """
    cli_runner(cli_kwargs, _params_, extract_roi_voxels )

import SimpleITK as sitk
import radiomics 
from pathlib import Path
import numpy as np
import pandas as pd

def extract_roi_voxels(input_itk_volume, input_itk_labels, resample_pixel_spacing, output_dir):
    file_stem = Path(input_itk_volume).stem

    sitk_image  = sitk.ReadImage(input_itk_volume)
    sitk_label  = sitk.ReadImage(input_itk_labels)

    sitk_image              = radiomics.imageoperations.normalizeImage(sitk_image)
    sitk_image, sitk_label  = radiomics.imageoperations.resampleImage (sitk_image, sitk_label, interpolator=sitk.sitkBSpline, resampledPixelSpacing=[resample_pixel_spacing, resample_pixel_spacing, resample_pixel_spacing], padDistance=20)

    output_itk_roi_image_volume = f'{output_dir}/roi_image_{file_stem}.nii'
    output_itk_roi_label_volume = f'{output_dir}/roi_label_{file_stem}.nii'
    output_npy_roi_image_volume = f'{output_dir}/roi_image_{file_stem}.npy'
    output_npy_roi_label_volume = f'{output_dir}/roi_label_{file_stem}.npy'

    output_feature_file = f'{output_dir}/feauture_data_{file_stem}.parquet'

    volume_cm3 =  (((sitk.GetArrayFromImage(sitk_label) > 0).sum()) * resample_pixel_spacing**3) / 1000.0

    sitk.WriteImage(sitk_image, output_itk_roi_image_volume)
    sitk.WriteImage(sitk_label,  output_itk_roi_label_volume)

    np.save(output_npy_roi_image_volume, sitk.GetArrayFromImage(sitk_image))
    np.save(output_npy_roi_label_volume, sitk.GetArrayFromImage(sitk_label))

    pd.DataFrame([{
        "input_itk_volume":input_itk_volume,
        "input_itk_labels":input_itk_labels,
	    "voxel_spacing": 1,
        "output_itk_roi_image_volume": output_itk_roi_image_volume,
        "output_itk_roi_label_volume": output_itk_roi_label_volume,
        "output_npy_roi_image_volume": output_npy_roi_image_volume,
        "output_npy_roi_label_volume": output_npy_roi_label_volume,
        "labeled_volume_cm3": volume_cm3,
    }]).to_parquet(output_feature_file)

    return {"feature_data": output_feature_file}

if __name__ == "__main__":
    cli()
