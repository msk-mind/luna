import numpy as np
import os
import pandas as pd
from dask.distributed import as_completed

from data_processing.common.dask import dask_job 
from data_processing.pathology.common.utils import get_slide_roi_masks, get_stain_vectors_macenko, extract_patch_texture_features

from scipy import stats
import pyarrow.parquet as pq
import pyarrow as pa

import logging
logger = logging.getLogger("extract_slide_texture_features")

@dask_job("stain_glcm")
def extract_slide_texture_features(index, output_dir, output_segment, slide_path, halo_roi_path, method_data):
    print ("Hello from extract_slide_texture_features()")
  
    annotation_name, stain_channel, TILE_SIZE = method_data['annotationLabel'], method_data['stainChannel'], method_data['tileSize']

    scale_factor = method_data.get("scaleFactor", None)
    glcm_feature = method_data.get("glcmFeature", "ClusterProminence")

    if (os.path.exists(f"{output_dir}/vector.npy")): 
        print (f"Output already generated {output_dir}/vector.npy, not doing anything...")
        features = np.load(f"{output_dir}/vector.npy")
    else:
        print ("Generating GLCM distribution...")

        features = np.array([])

        img_arr, sample_arr, mask_arr = get_slide_roi_masks(
            slide_path=slide_path, 
            halo_roi_path=halo_roi_path,
            annotation_name=annotation_name,
            scale_factor=scale_factor,
	    output_dir=output_dir) 

        vectors = get_stain_vectors_macenko(sample_arr)

        logger.info ("Stain vectors=", vectors)
        logger.info ("Max x levels:", img_arr.shape[0])

        nrow = 0
        for x in range(0, img_arr.shape[0], TILE_SIZE):
            nrow += 1
            for y in range(0, img_arr.shape[1], TILE_SIZE):
                
                img_patch  = img_arr [x:x+TILE_SIZE, y:y+TILE_SIZE, :]  
                mask_patch = mask_arr[x:x+TILE_SIZE, y:y+TILE_SIZE]

                if mask_patch.sum() == 0: continue
                
                address = f"{index}_{x}_{y}"
                print (f"Processing {address} @ {output_segment}")
                    
                try:
                    texture_values = extract_patch_texture_features(address, img_patch, mask_patch, stain_vectors=vectors, stain_channel=stain_channel, glcm_feature=glcm_feature)
    
                    if not texture_values is None:
                        features = np.append(features, texture_values)
                except Exception as exc:
                    print (f"Skipped tile {address} because: {exc}")
            print (f"On row {nrow} of {len(range(0, img_arr.shape[0], TILE_SIZE))}")
        
        np.save(f"{output_dir}/vector.npy", features)

    n, (smin, smax), sm, sv, ss, sk = stats.describe(features)
    ln_params = stats.lognorm.fit(features, floc=0)

    hist_features = {
        f'main_index': index,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_nobs': n,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_min': smin,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_max': smax,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_mean': sm,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_variance': sv,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_skewness': ss,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_kurtosis': sk,
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_lognorm_fit_p0': ln_params[0],
        f'pixel_original_glcm_{glcm_feature}_scale_{scale_factor}_channel_{stain_channel}_lognorm_fit_p2': ln_params[2]
    }

    data_table = pd.DataFrame(data=hist_features, index=[0]).set_index('main_index')
    pq.write_table(pa.Table.from_pandas(data_table), output_segment)

    print ("Saved to", output_segment)
    print (data_table)

    return output_dir, output_segment 

            
