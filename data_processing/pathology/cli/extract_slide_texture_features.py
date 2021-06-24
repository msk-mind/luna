import numpy as np
import os
from dask.distributed import as_completed

from data_processing.common.dask import with_dask_runner
from data_processing.pathology.common.utils import get_slide_roi_masks, get_stain_vectors_macenko, extract_patch_texture_features
from distributed import worker_client

@dask_job("stain_glcm")
def extract_slide_texture_features(index, output_dir, output_segment, slide_path, halo_roi_path, annotation_name, stain_channel, TILE_SIZE=500):
    print ("Hello from extract_slide_texture_features()")

    img_arr, sample_arr, mask_arr = get_slide_roi_masks(
        slide_path=slide_path, 
        halo_roi_path=halo_roi_path,
        annotation_name=annotation_name) 

    vectors = get_stain_vectors_macenko(sample_arr)

    print ("Stain vectors=", vectors)
    print ("Max x levels:", img_arr.shape[0])

    with worker_client() as runner:

        futures = []

        for x in range(0, img_arr.shape[0], TILE_SIZE):
            for y in range(0, img_arr.shape[1], TILE_SIZE):
                
                img_patch  = img_arr [x:x+TILE_SIZE, y:y+TILE_SIZE, :]  
                mask_patch = mask_arr[x:x+TILE_SIZE, y:y+TILE_SIZE]

                if mask_patch.sum() == 0: continue
                
                # Use random (instead of deterministic) hashes for result key (?)
                img_patch_future  = runner.scatter(img_patch)
                mask_patch_future = runner.scatter(mask_patch)

                address = f"{index}_{x}_{y}"
                
                futures.append (
                    runner.submit(extract_patch_texture_features, address, img_patch_future, mask_patch_future, stain_vectors=vectors, stain_channel=stain_channel, glcm_feature='original_glcm_ClusterTendency')
                )        

        features = np.array([])

        for future in as_completed(futures):
            
            try:
                texture_values = future.result()

                if not texture_values is None:
                    features = np.append(features, texture_values)
            except Exception as exc:
                print (f"Skipped future: {exc}")

    dest_dir=f"/gpfs/mskmind_ess/aukermaa/data/{index}/original_glcm_ClusterTendency/"
    os.makedirs(dest_dir, exist_ok=True)
    np.save(f"{dest_dir}/vector.npy", features)

    return dest_dir, features.mean()

            
