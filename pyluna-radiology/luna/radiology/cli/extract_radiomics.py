import os, json, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('extract_radiomics')

from luna.common.utils import cli_runner

from typing import List
_params_ = [('input_itk_volume', str), ('input_itk_labels', str), ('lesion_indicies', List[int]), ('output_dir', str), ('pyradiomics_config', dict), ('check_geometry_strict', bool), ('enable_all_filters', bool)]

@click.command()
@click.argument('input_itk_volume', nargs=1)
@click.argument('input_itk_labels', nargs=1)
@click.option('-idx', '--lesion_indicies', required=False,
              help='lesion labels given as a comma-separated list')
@click.option('-rcfg', '--pyradiomics_config', required=False,
              help='radiomic feature extractor parameters in json format')
@click.option('--check_geometry_strict', is_flag=True,
              help='enfource strictly matching geometries')
@click.option('--enable_all_filters', is_flag=True,
              help='enable all image filters automatically')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Extract radiomics given and image, label to and output_dir, parameterized by params

    \b
    Inputs:
        input_itk_volume: itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        input_itk_labels: itk compatible label volume (.mha, .nii, etc.)
    \b
    Outpus:
        feature_csv
    \b
    Example:
        extract_radiomics ct_image.mhd, ct_lesions.mha
            --lesion_indicies 1,2,3
            --pyradiomics_config '{"interpolator": "sitkBSpline","resampledPixelSpacing":[2.5,2.5,2.5]}'
            --enable_all_filters
			-o ./radiomics_result/
    """
    cli_runner(cli_kwargs, _params_, extract_radiomics_multiple_labels )


import medpy.io 
from pathlib import Path
import numpy as np
import pandas as pd
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

def extract_radiomics_multiple_labels(input_itk_volume: str, input_itk_labels: str, lesion_indicies: List[int], pyradiomics_config: dict, check_geometry_strict: bool, enable_all_filters: bool, output_dir: str):
    """Extract radiomics using pyradiomics, with additional checking of the input geometry match, resulting in a feature csv file

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        input_itk_labels (str): path to itk compatible label volume (.mha)
        output_dir (str): output/working directory
        lesion_indicies (List[int]): list of lesion indicies (label values) to process
        pyradiomics_config (dict): keyword arguments to pass to pyradiomics featureextractor
        check_geometry_strict (str): enforce strict match of the input geometries (otherwise, possible mismatch is silently ignored)
        enable_all_filters (bool): turns on all image filters

    Returns:
        dict: metadata about function call
    """
    
    image, image_header = medpy.io.load(input_itk_volume)

    if   Path(input_itk_labels).is_dir():  label_path_list = [str(path) for path in Path(input_itk_labels).glob("*")]
    elif Path(input_itk_labels).is_file(): label_path_list = [input_itk_labels]
    else: raise RuntimeError("Issue with detecting label format")

    available_labels = set()
    for label_path in label_path_list:
        label, label_header = medpy.io.load(label_path)

        available_labels.update

        available_labels.update(np.unique(label))

        logger.info(f"Checking {label_path}")

        if check_geometry_strict and not image_header.get_voxel_spacing() == label_header.get_voxel_spacing(): 
            raise RuntimeError(f"Voxel spacing mismatch, image.spacing={image_header.get_voxel_spacing()}, label.spacing={label_header.get_voxel_spacing()}" )
        
        if not image.shape == label.shape:
            raise RuntimeError(f"Shape mismatch: image.shape={image.shape}, label.shape={label.shape}")

    df_result = pd.DataFrame()

    for lesion_index in available_labels.intersection(lesion_indicies):

        extractor = featureextractor.RadiomicsFeatureExtractor(label=lesion_index, **pyradiomics_config)

        if enable_all_filters: extractor.enableAllImageTypes()

        for label_path in label_path_list:

            result = extractor.execute(input_itk_volume, label_path)

            result['lesion_index'] = lesion_index 

            df_result = pd.concat([df_result, pd.Series(result).to_frame()], axis=1)

    output_filename = os.path.join(output_dir, "radiomics.csv")

    df_result.T.to_csv(output_filename, index=False)

    logger.info(df_result.T)

    properties = {
        'feature_csv' : output_filename,
        'lesion_indicies': lesion_indicies,
    }

    return properties


if __name__ == "__main__":
    cli()
