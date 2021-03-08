'''
@author: aukermaa@mskcc.org
@author: pateld6@mskcc.org
@author: rosed2@mskcc.org

Various utility and processing methods for pathology
'''

import os, itertools, logging, re

import numpy  as np
import pandas as pd
import seaborn as sns

from PIL import Image

import openslide
from openslide.deepzoom import DeepZoomGenerator

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
from skimage.draw import rectangle_perimeter, rectangle

NUM_COLORS = 100 + 1
scoring_palette = sns.color_palette("viridis_r", n_colors=NUM_COLORS)
scoring_palette_as_list = [[int(x * 255) for x in scoring_palette.pop()] for i in range(NUM_COLORS)]


# USED -> utils
def array_to_slide(arr):
    assert isinstance(arr, np.ndarray)
    slide = openslide.ImageSlide(Image.fromarray(arr))
    return slide
    
# USED -> utils
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

# USED -> utils
def get_full_resolution_generator(slide, tile_size, level_offset=0):
    assert isinstance(slide, openslide.OpenSlide) or isinstance(slide, openslide.ImageSlide)
    generator = DeepZoomGenerator(slide, overlap=0, tile_size=tile_size, limit_bounds=False)
    generator_level = generator.level_count - 1 - level_offset
    if level_offset == 0:
        assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level

# USED -> generate cli
def get_otsu_scores(address_raster, otsu_img, otsu_tile_size):
    otsu_slide = array_to_slide(otsu_img)
    otsu_generator, otsu_generator_level = get_full_resolution_generator(otsu_slide, tile_size=otsu_tile_size)   
    otsu_score_results = []
    for address in address_raster:
        otsu_tile = np.array(otsu_generator.get_tile(otsu_generator_level, address))
        otsu_score_results.append( otsu_tile.mean().item() )
    return otsu_score_results

# USED -> generate cli
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

# USED -> utils
def coord_to_address(s, magnification):
    x = s[0]
    y = s[1]
    return f"x{x}_y{y}_z{magnification}"

# USED -> utils
def address_to_coord(s):
    p = re.compile('x(\d+)_y(\d+)_z(\d+)', re.IGNORECASE)
    m = p.match(s)
    x = int(m.group(1))
    y = int(m.group(2))
    return (x,y)

# USED -> utils
def get_downscaled_thumbnail(slide, scale_factor):
    new_width  = slide.dimensions[0] // scale_factor
    new_height = slide.dimensions[1] // scale_factor
    img = slide.get_thumbnail((new_width, new_height))
    return np.array(img)

# USED -> generate tiles
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

# USED -> vis tiles
def visualize_tiling_scores(df, thumbnail_img, tile_size):
    """
    Draw colored boxes around tiles 
    :param _thumbnail: np.ndarray
    :param tile_size: int
    :param tile_addresses:
    :return: new thumbnail image with black boxes around tiles passing threshold
    """

    assert isinstance(thumbnail_img, np.ndarray) and isinstance(tile_size, int)
    thumbnail = array_to_slide(thumbnail_img)
    generator, generator_level = get_full_resolution_generator(thumbnail, tile_size=tile_size)

    for index, row in df.iterrows():
        address = address_to_coord(str(index))
        extent = generator.get_tile_dimensions(generator_level, address)
        start = (address[1] * tile_size, address[0] * tile_size)  # flip because OpenSlide uses
                                                                    # (column, row), but skimage
                                                                    # uses (row, column)
        rr, cc = rectangle_perimeter(start=start, extent=extent, shape=thumbnail_img.shape)
        
        # set color based on intensity of value instead of black border (1)
        scaled_score = round(row.otsu_score * (NUM_COLORS-1))
        thumbnail_img[rr, cc] = scoring_palette_as_list[scaled_score]
    
    return thumbnail_img

### MAIN ENTRY METHOD -> pretile
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

    logger.info("Displaying DataFrame for otsu_score > 0.5:")
    logger.info (df [ df["otsu_score"] > 0.5 ])

    output_file = os.path.join(output_dir, "tile_scores_and_labels.csv")

    df.to_csv(output_file)

    logger.info ("Saved tile scores at %s", output_file)

    properties = {
        "file": output_file,
        "total_tiles": len(df),
        "available_labels": list(df.columns)
    }

    return properties


### MAIN ENTRY METHOD -> vis tiles
def visualize_scoring(slide_file_path: str, scores_file_path: str, output_dir: str, params: dict):
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
    output_file = os.path.join(output_dir, "tile_scores_and_labels_visualization.png")

    # Create thumbnail image for scoring
    rbg_thumbnail  = get_downscaled_thumbnail(slide, scale_factor)
    df_scores      = pd.read_csv(scores_file_path).set_index("address")

    thumbnail_overlayed = visualize_tiling_scores(df_scores, rbg_thumbnail, tile_size // scale_factor)
    thumbnail_overlayed = Image.fromarray(thumbnail_overlayed)
    thumbnail_overlayed.save(output_file)

    logger.info ("Saved visualization at %s", output_file)

    properties = {
        "file": output_file,
    }

    return properties