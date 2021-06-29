import numpy as np
import os, shutil
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from data_processing.common.dask import dask_job 
from data_processing.common.custom_logger import init_logger
from data_processing.radiology.common.preprocess import window_dicoms, generate_scan, randomize_contours, extract_radiomics
from distributed import worker_client

init_logger()

@dask_job('recist_radiomics')
def extract_recist_radiomics(index, output_dir, output_segment, dicom_path, segment_path, method_data):
    """ 
    The RECIST-style radiomics job, consisting of multiple task modules 
    """

    shutil.copy(segment_path, output_dir)

    windowed_CT_dicoms_properties = window_dicoms (
        dicom_path = dicom_path, 
        output_dir = output_dir, 
        params = method_data, 
        tag="windowed_CT_dicoms" )
    
    source_CT_nii_properties = generate_scan (
        dicom_path = dicom_path, 
        output_dir = output_dir, 
        params = method_data, 
        tag="source_CT_nii")

    windowed_CT_nii_properties = generate_scan (
        dicom_path = windowed_CT_dicoms_properties['data'], 
        output_dir = output_dir, 
        params = method_data, 
        tag="windowed_CT_nii")

    if method_data.get("enableMirp", False):
         image_properties, label_properties, pertubation_properties, supervoxel_properties = randomize_contours(
             image_path = windowed_CT_nii_properties['data'], 
             label_path = segment_path, 
             output_dir = output_dir, 
             params = method_data, 
             tag='randomized_segments')

    radiomics_results = []

    for label in [1,2,3,4,5,6]:
        method_data['radiomicsFeatureExtractor']['label'] = label

        result = extract_radiomics(
            image_path = windowed_CT_nii_properties['data'], 
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

    pq.write_table(pa.Table.from_pandas(data_table), output_segment)

