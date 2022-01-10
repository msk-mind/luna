# General imports
import os, json, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('window_volume')

from luna.common.utils import cli_runner

from typing import List
_params_ = [('input_image_data', str), ('input_label_data', str), ('lesion_indicies', List[int]), ('output_dir', str), ('pyradiomics_config', json), ('check_geometry_strict', bool), ('enable_all_filters', bool)]

@click.command()
@click.option('-ii', '--input_image_data', required=False,
              help='path to input image data')
@click.option('-il', '--input_label_data', required=False,
              help='path to input label data')
@click.option('-idx', '--lesion_indicies', required=False,
              help='path to input label data')
@click.option('-rcfg', '--pyradiomics_config', required=False,
              help='path to input label data')
@click.option('--check_geometry_strict', is_flag=True,
              help='')
@click.option('--enable_all_filters', is_flag=True,
              help='')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Run with explicit arguments:

    \b
        infer_tiles
            -i 1412934/data/TileImages
            -o 1412934/data/TilePredictions
            -rn msk-mind/luna-ml:main 
            -tn tissue_tile_net_transform 
            -mn tissue_tile_net_model_5_class
            -wt main:tissue_net_2021-01-19_21.05.24-e17.pth

    Run with implicit arguments:

    \b
        infer_tiles -m 1412934/data/TilePredictions/metadata.json
    
    Run with mixed arguments (CLI args override yaml/json arguments):

    \b
        infer_tiles --input_data 1412934/data/TileImages -m 1412934/data/TilePredictions/metadata.json
    """
    cli_runner(cli_kwargs, _params_, extract_radiomics_multiple_labels )


import medpy.io 
from pathlib import Path
import numpy as np
import pandas as pd
from radiomics import featureextractor  # This module is used for interaction with pyradiomics

def extract_radiomics_multiple_labels(input_image_data, input_label_data, lesion_indicies, pyradiomics_config, check_geometry_strict, enable_all_filters, output_dir):
        """
        Extract radiomics given and image, label to and output_dir, parameterized by params

        :param image_path: filepath to image
        :param label_path: filepath to 3d segmentation(s) as single path or list
        :param output_dir: destination directory
        :param params {
            radiomicsFeatureExtractor dict: configuration for the RadiomicsFeatureExtractor
            enableAllImageTypes bool: flag to enable all image types
        }

        :return: property dict, None if function fails
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image, image_header = medpy.io.load(input_image_data)

        if   Path(input_label_data).is_dir():  label_path_list = [str(path) for path in Path(input_label_data).glob("*")]
        elif Path(input_label_data).is_file(): label_path_list = [input_label_data]
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

                result = extractor.execute(input_image_data, label_path)

                result['lesion_index'] = lesion_index 

                df_result = pd.concat([df_result, pd.Series(result).to_frame()], axis=1)

        output_filename = os.path.join(output_dir, "radiomics.csv")

        df_result.T.to_csv(output_filename, index=False)

        logger.info(df_result.T)

        properties = {
            'data' : output_filename,
            'lesion_indicies': lesion_indicies,
        }

        return properties


if __name__ == "__main__":
    cli()
