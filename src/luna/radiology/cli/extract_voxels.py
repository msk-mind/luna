import os
from pathlib import Path

import fire
import medpy.io
import numpy as np
from loguru import logger



def extract_voxels(input_itk_volume, output_dir):
    """Save a numpy file from a given ITK volume

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """
    file_stem = Path(input_itk_volume).stem
    # file_ext = Path(input_itk_volume).suffix

    outFileName = os.path.join(output_dir, file_stem + ".npy")

    image, header = medpy.io.load(input_itk_volume)

    np.save(outFileName, image)

    logger.info(f"Extracted voxels of shape {image.shape}")

    # Prepare metadata and commit
    properties = {
        "npy_volume": outFileName,
    }

    return properties


if __name__ == "__main__":
    fire.Fire(extract_voxels)
