import numpy as np
import os
import pandas as pd

from luna.common.dask import dask_job
from luna.pathology.common.utils import get_slide_roi_masks, get_stain_vectors_macenko, extract_patch_texture_features

from scipy import stats
import pyarrow.parquet as pq
import pyarrow as pa

@dask_job("stain_glcm")
def extract_slide_texture_features(index, output_segment, slide_path, halo_roi_path, method_data):
    """Extract slide texture features

    Args:
        index (string): main index string
        output_segment (string): path to write result parquet
        slide_path (string): path to the whole slide image
        halo_roi_path (string): path to halo roi path
        method_data (dict): method parameters with annotation and tile details 
            including annotationLabel, stainChannel and tileSize

    Returns:
        tuple: path to features saved as a np.array & path to feature metadata saved as a parquet.
    """
    print ("Hello from extract_slide_texture_features()")

    annotation_name, stain_channel, TILE_SIZE = method_data['annotationLabel'], method_data['stainChannel'], method_data['tileSize']

    dest_dir=f"/gpfs/mskmind_ess/aukermaa/data/{index}/original_glcm_ClusterTendency/"
    os.makedirs(dest_dir, exist_ok=True)

    img_arr, sample_arr, mask_arr = get_slide_roi_masks(
        slide_path=slide_path,
        halo_roi_path=halo_roi_path,
        annotation_name=annotation_name)

    vectors = get_stain_vectors_macenko(sample_arr)

    print ("Stain vectors=", vectors)
    print ("Max x levels:", img_arr.shape[0])

    if (os.path.exists(f"{dest_dir}/vector.npy")):
        print ("Output already generated, not doing anything...")
        return dest_dir, output_segment

    features = np.array([])

    nrow = 0
    for x in range(0, img_arr.shape[0], TILE_SIZE):
        nrow += 1
        for y in range(0, img_arr.shape[1], TILE_SIZE):

            img_patch  = img_arr [x:x+TILE_SIZE, y:y+TILE_SIZE, :]
            mask_patch = mask_arr[x:x+TILE_SIZE, y:y+TILE_SIZE]

            if mask_patch.sum() == 0: continue

            address = f"{index}_{x}_{y}"

            try:
                texture_values = extract_patch_texture_features(address,
                                                                img_patch,
                                                                mask_patch,
                                                                stain_vectors=vectors,
                                                                stain_channel=stain_channel,
                                                                glcm_feature='original_glcm_ClusterTendency')

                if not texture_values is None:
                    features = np.append(features, texture_values)
            except Exception as exc:
                print (f"Skipped tile {address} because: {exc}")
        print (f"On row {nrow} of {len(range(0, img_arr.shape[0], TILE_SIZE))}")

    n, (smin, smax), sm, sv, ss, sk = stats.describe(features)
    hist_features = {
        'main_index': index,
        'pixel_original_glcm_ClusterTendency_nobs': n,
        'pixel_original_glcm_ClusterTendency_min': smin,
        'pixel_original_glcm_ClusterTendency_max': smax,
        'pixel_original_glcm_ClusterTendency_mean': sm,
        'pixel_original_glcm_ClusterTendency_variance': sv,
        'pixel_original_glcm_ClusterTendency_skewness': ss,
        'pixel_original_glcm_ClusterTendency_kurtosis': sk
    }

    data_table = pd.DataFrame(data=hist_features, index=[0]).set_index('main_index')
    print (data_table)
    pq.write_table(pa.Table.from_pandas(data_table), output_segment)
    print ("Saved to", output_segment)
    np.save(f"{dest_dir}/vector.npy", features)

    return dest_dir, output_segment


