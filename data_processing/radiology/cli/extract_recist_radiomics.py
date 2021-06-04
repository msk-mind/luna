import numpy as np
import os, shutil
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from data_processing.common.dask import with_dask_runner
from data_processing.common.custom_logger import init_logger
from data_processing.radiology.common.preprocess import window_dicoms, generate_scan, randomize_contours, extract_radiomics

init_logger()

@with_dask_runner
def extract_recist_radiomics(namespace, scan_id, dicom_path, segment_path, runner=None):

    method_data = {
        "window": True,
        "windowLowLevel": 0,
        "windowHighLevel": 200,
        "itkImageType": 'nii',
        "strictGeometry": False,
        "enableAllImageTypes": True,
        "mirpResampleSpacing": [1.25, 1.25, 1.25],
        "mirpResampleBeta": 0.95,
        "radiomicsFeatureExtractor": {
            "binWidth": 10,
            "resampledPixelSpacing": [1.25, 1.25, 1.25],
            "verbose": "False",
            "geometryTolerance":1e-08
        }
    }

    output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, scan_id)
    ouput_ds   = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, "EXTRACT_RECIST_DS")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ouput_ds,   exist_ok=True)

    shutil.copy(segment_path, output_dir)

    windowed_CT_dicoms_properties = window_dicoms (
        dicom_path = dicom_path, 
        output_dir = output_dir, 
        params = method_data, 
        tag="windowed_CT_dicoms" )

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
    for label in [1,2,3,4,5,6]:
        method_data['radiomicsFeatureExtractor']['label'] = label

        future = runner.submit (extract_radiomics, 
            image_path = windowed_CT_nii_properties['data'], 
            label_path = segment_path, 
            output_dir = output_dir, 
            params = method_data, 
            tag=f'unfiltered-radiomics-label-{label}')
        radiomics_futures.append(future)

        future = runner.submit (extract_radiomics, 
            image_path = image_properties['data'], 
            label_path = label_properties['data'], 
            output_dir = output_dir, 
            params = method_data, 
            tag=f'filtered-radiomics-label-{label}')
        radiomics_futures.append(future)

        future = runner.submit (extract_radiomics, 
            image_path = image_properties['data'], 
            label_path = pertubation_properties['data'], 
            output_dir = output_dir, 
            params = method_data, 
            tag=f'pertubation-radiomics-label-{label}')
        radiomics_futures.append(future)

    data_table =  pd.concat(runner.gather(radiomics_futures) ).set_index('lesion_id')

    data_table = data_table.loc[:, ~data_table.columns.str.contains('diag')]

    data_table = data_table.astype(float)

    print (data_table)

    output_slice = os.path.join(ouput_ds, f"ResultSegment-{scan_id}.parquet")

    pq.write_table(pa.Table.from_pandas(data_table), output_slice)


if __name__=='__main__':
    from dask.distributed import Client
    from dask.distributed import wait

    
    client = Client(threads_per_worker=1, n_workers=20)

    futures = []
    futures.append( client.submit (extract_recist_radiomics, "testing", "347911", "/mskmind_lung/347911/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/123753 segments/347911_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "348021", "/mskmind_lung/348021/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/115159 segments/348021_NH_UPDATED.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347917", "/mskmind_lung/347917/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/152504 segments/347917_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "348008", "/mskmind_lung/348008/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/111922 segments/348008_NH.mha") )

    futures.append( client.submit (extract_recist_radiomics, "testing", "190323", "/mskmind_lung/190323/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/190323_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "190400", "/mskmind_lung/190400/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/190400_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "190960", "/mskmind_lung/190960/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/190960_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347883", "/mskmind_lung/347883/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347883_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347897", "/mskmind_lung/347897/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347897_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347916", "/mskmind_lung/347916/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347916_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347953", "/mskmind_lung/347953/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347953_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347960", "/mskmind_lung/347960/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347960_AMP.mha") )
    futures.append( client.submit (extract_recist_radiomics, "testing", "347981", "/mskmind_lung/347981/SCANS/2/DICOM/", "/gpfs/mskmind_ess/vangurir/AMP redo segments/347981_AMP.mha") )


    wait (futures)

