import numpy as np
import os, shutil
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from luna.common.wrapper import service 
from luna.radiology.common.preprocess import window_volume, generate_scan, randomize_contours, extract_radiomics

@service('recist_radiomics')
def extract_recist_radiomics(index, output_dir, dicom_path, segment_path, method_data):
    """ 
    The RECIST-style radiomics job, consisting of multiple task modules 
    """

    shutil.copy(segment_path, output_dir)
    
    source_CT_nii_properties = generate_scan (
        dicom_path = dicom_path, 
        output_dir = output_dir, 
        params = method_data, 
        tag="source_CT_nii")

    source_CT_nii_properties = window_volume(
        image_path = source_CT_nii_properties['data'], 
        output_dir = output_dir, 
        params = method_data, 
        tag="windowed_CT_nii")

    if method_data.get("enableMirp", False):
         image_properties, label_properties, pertubation_properties, supervoxel_properties = randomize_contours(
             image_path = source_CT_nii_properties['data'], 
             label_path = segment_path, 
             output_dir = output_dir, 
             params = method_data, 
             tag='randomized_segments')

    radiomics_results = []

    for label in method_data['lesionLabels']: 
        method_data['radiomicsFeatureExtractor']['label'] = label

        result = extract_radiomics(
            image_path = source_CT_nii_properties['data'], 
            label_path = segment_path, 
            output_dir = output_dir, 
            params = method_data, 
            tag=f'unfiltered-radiomics')
        radiomics_results.append( result )

        if method_data.get("enableMirp", False):
            result = extract_radiomics(
                image_path = image_properties['data'], 
                label_path = label_properties['data'], 
                output_dir = output_dir, 
                params = method_data, 
                tag=f'filtered-radiomics')
            radiomics_results.append( result )
    
            result = extract_radiomics(
                image_path = image_properties['data'], 
                label_path = pertubation_properties['data'], 
                output_dir = output_dir, 
                params = method_data, 
                tag=f'pertubation-radiomics')
            radiomics_results.append( result )

    data_table =  pd.concat(radiomics_results)
    data_table = data_table.loc[:, ~data_table.columns.str.contains('diag')]
    data_table = data_table.astype(float)

    data_table['main_index'] = index
    data_table = data_table.reset_index().set_index(['main_index','job_tag', 'lesion_index'])
    return data_table
    # pq.write_table(pa.Table.from_pandas(data_table), output_segment)

