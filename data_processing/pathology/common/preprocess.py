'''
@author: aukermaa@mskcc.org
@author: pateld6@mskcc.org
@author: rosed2@mskcc.org

Various utility and processing methods for pathology
'''

import os, itertools, logging

import numpy  as np
import pandas as pd

from PIL import Image

import openslide
from openslide.deepzoom import DeepZoomGenerator

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu

# USED
def array_to_slide(arr):
    assert isinstance(arr, np.ndarray)
    slide = openslide.ImageSlide(Image.fromarray(arr))
    return slide
    
# USED
def get_scale_factor_at_magnfication(slide, requested_mangification):
    scanned_magnfication = int(slide.properties['aperio.AppMag'])  
    # verify magnification valid
    scale_factor = 1
    if scanned_magnfication != requested_mangification:
        if scanned_magnfication < requested_mangification:
            raise ValueError(f'Expected magnification >={requested_mangification} but got {scanned_magnfication}')
        elif (scanned_magnfication % requested_mangification) == 0:
            scale_factor = scanned_magnfication // requested_mangification
        else:
            raise ValueError(f'Expected magnification {requested_mangification} to be an divisor multiple of {scanned_magnfication}')
    return scale_factor

# USED
def get_full_resolution_generator(slide, tile_size, level_offset=0):
    assert isinstance(slide, openslide.OpenSlide) or isinstance(slide, openslide.ImageSlide)
    generator = DeepZoomGenerator(slide, overlap=0, tile_size=tile_size, limit_bounds=False)
    generator_level = generator.level_count - 1 - level_offset
    if level_offset == 0:
        assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level

# USED
def get_otsu_scores(address_raster, otsu_img, otsu_tile_size):
    otsu_slide = array_to_slide(otsu_img)
    otsu_generator, otsu_generator_level = get_full_resolution_generator(otsu_slide, tile_size=otsu_tile_size)   
    otsu_score_results = []
    for address in address_raster:
        otsu_tile = np.array(otsu_generator.get_tile(otsu_generator_level, address))
        otsu_score_results.append( otsu_tile.mean().item() )
    return otsu_score_results

# USED
def get_purple_scores(address_raster, rgb_img, rgb_tile_size):
    rgb_slide = array_to_slide(rgb_img)
    rgb_generator, rgb_generator_level = get_full_resolution_generator(rgb_slide, tile_size=rgb_tile_size)   
    purple_score_results = []
    for address in address_raster:
        rgb_tile = np.array(rgb_generator.get_tile(rgb_generator_level, address))
        r, g, b = rgb_tile[..., 0], rgb_tile[..., 1], rgb_tile[..., 2]
        # cond1 = r > 75
        # cond2 = b > 90
        # score = np.sum(cond1 & cond2)
        score = np.sum((r > (g + 10)) & (b > (g + 10))) / rgb_tile.size
        purple_score_results.append ( score )  
    return purple_score_results

# USED
def coord_to_address(s, magnification):
    x = s[0]
    y = s[1]
    return f"x{x}_y{y}_z{magnification}"

# USED
def get_downscaled_thumbnail(slide, scale_factor):
    new_width  = slide.dimensions[0] // scale_factor
    new_height = slide.dimensions[1] // scale_factor
    img = slide.get_thumbnail((new_width, new_height))
    return np.array(img)

# USED
def make_otsu(img, scale=1):
    """
    Make image with pixel-wise foreground/background labels.
    :param img: grayscale np.ndarray
    :return: np.ndarray where each pixel is 0 if background and 1 if foreground
    """
    assert isinstance(img, np.ndarray)
    _img = rgb2gray(img)
    threshold = threshold_otsu(_img)
    return (_img < (threshold * scale)).astype(float)

### MAIN ENTRY METHOD 
def pretile_scoring(slide_file_path: str, output_dir: str, params: dict):
    logger = logging.getLogger(__name__)

    tile_size       = params.get("tile_size")
    magnification   = params.get("magnification")
    scale_factor    = params.get("scale_factor", 16)

    logger.info("Processing slide %s", slide_file_path)
    logger.info("Params = %s", params)

    slide = openslide.OpenSlide(slide_file_path)

    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])
 
    mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_mangification=magnification)

    scale_factor *=  mag_scale_factor

    logger.info("Normalized mangification scale factor for %sx is %s, overall thumbnail scale factor is %s", magnification, mag_scale_factor, scale_factor)

    # Create thumbnail image for scoring
    rbg_thumbnail  = get_downscaled_thumbnail(slide, scale_factor)
    otsu_thumbnail = make_otsu(rbg_thumbnail)

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=tile_size, level_offset=0)

    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    
    address_raster = [{"address": coord_to_address(address, magnification), "coordinates": address} for address in itertools.product(range(tile_x_count), range(tile_y_count))]
    logger.info("Number of tiles in raster: %s", len(address_raster))

    df = pd.DataFrame(address_raster).set_index("address")

    df.loc[:, "otsu_score"  ] = get_otsu_scores   (df['coordinates'], otsu_thumbnail, tile_size // scale_factor)
    df.loc[:, "purple_score"] = get_purple_scores (df['coordinates'], rbg_thumbnail,  tile_size // scale_factor)

    logger.info (df [ df["otsu_score"] > 0.5 ])

    output_file = os.path.join(output_dir, "tile_scores_and_labels.csv")

    df.to_csv(output_file)

    logger.info ("Saved tile scores at %s", output_file)

    properties = {
        "path": output_dir,
        "total_tiles": len(df),
        "available_labels": list(df.columns)
    }

    return properties