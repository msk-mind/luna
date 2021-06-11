import numpy as np
import os, shutil
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from data_processing.common.dask import with_dask_runner
from data_processing.common.custom_logger import init_logger
from data_processing.radiology.common.preprocess import window_dicoms, generate_scan, randomize_contours, extract_radiomics
from distributed import worker_client

init_logger()

@with_dask_runner
def extract_recist_radiomics(namespace, indicies, dicom_path, segment_path):

    unique_key = indicies 

    method_data = {
        "window": True,
        "windowLowLevel": -1150,
        "windowHighLevel": 250,
        "itkImageType": 'nii',
        "strictGeometry": False,
        "enableAllImageTypes": False,
        "mirpResampleSpacing": [1.25, 1.25, 1.25],
        "mirpResampleBeta": 0.95,
        "radiomicsFeatureExtractor": {
            "binWidth": 10,
            "resampledPixelSpacing": [1.25, 1.25, 1.25],
            "verbose": "False",
            "geometryTolerance":1e-08
        }
    }

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, unique_key)
    ouput_ds   = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, "EXTRACT_RECIST_DS")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ouput_ds,   exist_ok=True)

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

    image_properties, label_properties, pertubation_properties, supervoxel_properties = randomize_contours(
        image_path = windowed_CT_nii_properties['data'], 
        label_path = segment_path, 
        output_dir = output_dir, 
        params = method_data, 
        tag='randomized_segments')

    radiomics_futures = []

    with worker_client() as runner:

        for label in [1,2,3,4,5,6]:
            method_data['radiomicsFeatureExtractor']['label'] = label

            future = runner.submit (extract_radiomics, 
                image_path = windowed_CT_nii_properties['data'], 
                label_path = segment_path, 
                output_dir = output_dir, 
                params = method_data, 
                tag=f'unfiltered-radiomics-label-{label}',
                priority=10,)
            radiomics_futures.append(future)

            future = runner.submit (extract_radiomics, 
                image_path = image_properties['data'], 
                label_path = label_properties['data'], 
                output_dir = output_dir, 
                params = method_data, 
                tag=f'filtered-radiomics-label-{label}',
                priority=10,)
            radiomics_futures.append(future)

            future = runner.submit (extract_radiomics, 
                image_path = image_properties['data'], 
                label_path = pertubation_properties['data'], 
                output_dir = output_dir, 
                params = method_data, 
                tag=f'pertubation-radiomics-label-{label}',
                priority=10,)
            radiomics_futures.append(future)

        all_results = runner.gather(radiomics_futures)

    data_table =  pd.concat(all_results).set_index('lesion_id')
    data_table = data_table.loc[:, ~data_table.columns.str.contains('diag')]
    data_table = data_table.astype(float)

    data_table['case_accession_number'] = unique_key

    output_slice = os.path.join(ouput_ds, f"ResultSegment-{unique_key}.parquet")

    pq.write_table(pa.Table.from_pandas(data_table), output_slice)


if __name__=='__main__':
    from dask.distributed import Client
    from dask.distributed import wait
    import pathlib
    import pandas as pd


    df = pd.read_csv("/gpfs/mskmindhdp_emc/user/shared_data_folder/lung-mind-project/inventory/inventory.csv").set_index("deid")
    futures = []

    client = Client(threads_per_worker=1, n_workers=16)

    for idx, row in df.iterrows():
        if row.has_radiology_segmentation==0: continue
        if row.thoracic_disease==0: continue

        print (idx, row.dicom_path, row.radiology_segmentation_path)

        futures.append( client.submit (extract_recist_radiomics, "LUNG_RADIOMICS_spacing1.25", str(idx), row.dicom_path, row.radiology_segmentation_path) )
       
    wait (futures)

