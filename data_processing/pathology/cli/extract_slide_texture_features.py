import numpy as np
from dask.distributed import as_completed

from data_processing.common.dask import job_runner
from data_processing.pathology.common.utils import get_slide_roi_masks, get_stain_vectors_macenko, extract_patch_texture_features

@job_runner
def extract_slide_texture_features(slide_path, halo_roi_path, annotation_name, stain_channel, TILE_SIZE=500, runner=None):

    img_arr, sample_arr, mask_arr = get_slide_roi_masks(
        slide_path=slide_path, 
        halo_roi_path=halo_roi_path,
        annotation_name=annotation_name) 

    vectors = get_stain_vectors_macenko(sample_arr)

    print ("Stain vectors=", vectors)
    print ("Max x levels:", img_arr.shape[0])

    futures = []

    for x in range(0, img_arr.shape[0], TILE_SIZE):
        for y in range(0, img_arr.shape[1], TILE_SIZE):
            
            img_patch  = img_arr [x:x+TILE_SIZE, y:y+TILE_SIZE, :]  
            mask_patch = mask_arr[x:x+TILE_SIZE, y:y+TILE_SIZE]
            
            img_patch_future  = runner.scatter(img_patch)
            mask_patch_future = runner.scatter(mask_patch)
            
            futures.append (
                runner.submit(extract_patch_texture_features, img_patch_future, mask_patch_future, stain_vectors=vectors, stain_channel=stain_channel, glcm_feature='original_glcm_ClusterTendency')
            )

    features = np.array([])

    for future in as_completed(futures):
        
        try:
            texture_values = future.result()

            if not texture_values is None:
                features = np.append(features, texture_values)
                print (f"Current mean: {features.mean()}")
        except Exception as exc:
            print (f"Skipped future: {exc}")

    np.save("./example_vector.npy", features)
    return "./example_vector.npy", features.mean()

            