import os
from pathlib import Path
from loguru import logger
import fire
import medpy.io
import numpy as np



def window_volume(
    input_itk_volume: str, output_dir: str, low_level: float, high_level: float
):
    """Applies a window function (clipping) to an input itk volume, outputs windowed volume

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        output_dir (str): output/working directory
        low_level (float): lower bound of clipping operation
        high_level (float): higher bound of clipping operation

    Returns:
        dict: metadata about function call
    """
    file_stem = Path(input_itk_volume).stem
    file_ext = Path(input_itk_volume).suffix

    outFileName = os.path.join(output_dir, file_stem + ".windowed" + file_ext)

    logger.info("Applying window [%s,%s]", low_level, high_level)

    image, header = medpy.io.load(input_itk_volume)
    image = np.clip(image, low_level, high_level)
    medpy.io.save(image, outFileName, header)
    # Prepare metadata and commit
    properties = {
        "itk_volume": outFileName,
    }

    return properties


if __name__ == "__main__":
    fire.Fire(window_volume)
