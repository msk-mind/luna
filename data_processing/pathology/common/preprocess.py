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

import json 
from pathlib import Path

from PIL import Image

import openslide
from openslide.deepzoom import DeepZoomGenerator

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
from skimage.draw import rectangle_perimeter, rectangle

import requests
import importlib

from shapely.geometry import shape, Point, Polygon

from random import randint
import torch


palette = sns.color_palette("viridis",as_cmap=True)
categorial = sns.color_palette("Set1", 8)
categorical_colors = {}

def get_tile_color(score):
    # categorical
    if isinstance(score, str):
        if score in categorical_colors:
            return categorical_colors[score]
        else:
            tile_color =  255 * np.array (categorial[len(categorical_colors.keys())])
            categorical_colors[score] = tile_color
            return tile_color

    # float, expected to be value from [0,1]
    elif isinstance(score, float) and score <= 1.0 and score >= 0.0:
        tile_color = [int(255*i) for i in palette(score)[:3]]
        return tile_color

    else:
        print("Invalid Score Type")
        return None


# USED -> utils
def array_to_slide(arr):
    assert isinstance(arr, np.ndarray)
    slide = openslide.ImageSlide(Image.fromarray(arr))
    return slide
    
# USED -> utils
def get_scale_factor_at_magnfication(slide, requested_magnification):
    """
    Return a scale factor if slide scanned magnification and
    requested magnification are different.
    :param slide: Openslide slide object
    :param requested_magnification: int
    :return: scale factor
    """
    
    # First convert to float to handle true integers encoded as string floats (e.g. '20.000')
    mag_value = float(slide.properties['aperio.AppMag'])

    # Then convert to integer
    scanned_magnfication = int (mag_value)

    # Make sure we don't have non-integer magnifications
    assert int (mag_value) == mag_value

    # Verify magnification valid
    scale_factor = 1
    if scanned_magnfication != requested_magnification:
        if scanned_magnfication < requested_magnification:
            raise ValueError(f'Expected magnification >={requested_magnification} but got {scanned_magnfication}')
        elif (scanned_magnfication % requested_magnification) == 0:
            scale_factor = scanned_magnfication // requested_magnification
        else:
            raise ValueError(f'Expected magnification {requested_magnification} to be an divisor multiple of {scanned_magnfication}')
    return scale_factor

# USED -> utils
def get_full_resolution_generator(slide, tile_size):
    assert isinstance(slide, openslide.OpenSlide) or isinstance(slide, openslide.ImageSlide)
    generator = DeepZoomGenerator(slide, overlap=0, tile_size=tile_size, limit_bounds=False)
    generator_level = generator.level_count - 1
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
        score = np.mean((r > (g + 10)) & (b > (g + 10)))
        purple_score_results.append ( score )  
    return purple_score_results

# USED -> utils
def coord_to_address(s, magnification):
    x = s[0]
    y = s[1]
    return f"x{x}_y{y}_z{magnification}"

# USED -> utils
def address_to_coord(s):
    s = str(s)
    p = re.compile('x(\d+)_y(\d+)_z(\d+)', re.IGNORECASE)
    m = p.match(s)
    x = int(m.group(1))
    y = int(m.group(2))
    return (x,y)

# USED -> utils
def get_downscaled_thumbnail(slide, scale_factor):
    """
    Return downscaled thumbnail
    :param slide: OpenSlide slide object
    :param scale_factor: scale factor
    :return: downscaled np.ndarray
    """
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
'''
def visualize_tiling_scores(df, thumbnail_img, tile_size, score_type_to_visualize):
    """
    Draw colored boxes around tiles 
    :param thumbnail_img: np.ndarray
    :param tile_size: int
    :param score_type_to_visualize: column name
    :return: new thumbnail image with black boxes around tiles passing threshold
    """

    assert isinstance(thumbnail_img, np.ndarray) and isinstance(tile_size, int)
    thumbnail = array_to_slide(thumbnail_img)
    generator, generator_level = get_full_resolution_generator(thumbnail, tile_size=tile_size)

    df_tiles_to_process = df[ (df["otsu_score"] > 0.5) &  (df["otsu_score"] > 0.1) ]

    for index, row in df_tiles_to_process.iterrows():
        address = address_to_coord(index)

        if 'regional_label' in row and pd.isna(row.regional_label): continue

        extent = generator.get_tile_dimensions(generator_level, address)
        start = (address[1] * tile_size, address[0] * tile_size)  # flip because OpenSlide uses
                                                                    # (column, row), but skimage
                                                                    # uses (row, column)
        rr, cc = rectangle_perimeter(start=start, extent=extent, shape=thumbnail_img.shape)
        
        # set color based on intensity of value instead of black border (1)
        score = row[score_type_to_visualize]
        thumbnail_img[rr, cc] = get_tile_color(score)
    
    return thumbnail_img
'''

def build_shapely_polygons_from_geojson(annotation_geojson):
    """
    Build shapely polygons from geojson

    :param annotation_geojson: geojson
    :return: polygon and annotation label lists
    """
    annotation_polygons = []
    annotation_labels = []
    for feature in annotation_geojson['features']:

        coords = feature['geometry']['coordinates']
        class_name = feature['properties']['label_name']

        # filter out lines that may be present (polygons containing 2 coordinates)
        if len(coords) >= 3:
            annotation_polygon = Polygon(coords)
            annotation_polygons.append(annotation_polygon)
            annotation_labels.append(class_name)
    return annotation_polygons,annotation_labels


def get_regional_labels(address_raster, annotation_polygons, annotation_labels, full_generator, full_level):
    """
    Return annotation labels for tiles that contain annotations
    If the tile doesn't contain annotation, set label to None.

    :param address_raster: coordinates
    :param annotation_polygons: shapely Polygon
    :param annotation_labels: annotation label name
    :param full_generator: full res generator
    :param full_level: full res level
    :return: annotation labels
    """
    regional_label_results = []

    for address in address_raster:
        tile_contains_annotation = False
        tile,_,tile_size = full_generator.get_tile_coordinates(full_level, address)

        tile_x, tile_y = tile
        tile_size_x, tile_size_y = tile_size

        tile_polygon = Polygon([
            (tile_x,               tile_y),
            (tile_x,               tile_y+tile_size_y),
            (tile_x+tile_size_x,   tile_size_y + tile_y),
            (tile_x + tile_size_x, tile_y),
            ])

        for annotation_polygon, annotation_label in zip(annotation_polygons,annotation_labels):
            if annotation_polygon.contains(tile_polygon):
                tile_contains_annotation = True
                regional_label_results.append (annotation_label)
                break
        if not tile_contains_annotation:
            regional_label_results.append (None)

    return regional_label_results

### MAIN ENTRY METHOD -> pretile
def pretile_scoring(slide_file_path: str, output_dir: str, params: dict, image_id: str):
    """
    Generate tiles and scores.

    Notes: 
    to_mag_scale_factor tells us how much to scale to get from full resolution to desired magnification
    to_thumbnail_scale_factor tells us how much to scale to get from desired magnification to the desired thumbnail downscale, relative to requested mag
    
    The tile size is defined at the requested mag, so it's bigger at full resolution and smaller for the thumbnail
    to_mag_scale_factor and to_thumbnail_scale_factor both need to be event integers, i.e. the scale factors are multiples of the the scanned magnficiation
    """
    logger = logging.getLogger(__name__)

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("magnification")

    # non-required arguments related to slideviewer annotations
    slideviewer_dmt           = params.get("slideviewer_dmt", None)
    labelset                  = params.get("labelset", None)

    logger.info("Processing slide %s", slide_file_path)
    logger.info("Params = %s", params)

    slide = openslide.OpenSlide(str(slide_file_path))

    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])
 
    scale_factor = params.get("scale_factor", 4) # Instead, we should specifiy a thumbnail zoom, and calculate this using get_scale_factor_at_magnfication()
    to_mag_scale_factor         = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)
    to_thumbnail_scale_factor   = to_mag_scale_factor * scale_factor

    if not to_mag_scale_factor % 1 == 0 or not requested_tile_size % scale_factor == 0: 
        raise ValueError("You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size = requested_tile_size * to_mag_scale_factor
    thumbnail_tile_size       = requested_tile_size // scale_factor

    logger.info("Normalized magnification scale factor for %sx is %s, overall thumbnail scale factor is %s", requested_magnification, to_mag_scale_factor, to_thumbnail_scale_factor)
    logger.info("Requested tile size=%s, tile size at full magnficiation=%s, tile size at thumbnail=%s", requested_tile_size, full_resolution_tile_size, thumbnail_tile_size)

    # Create thumbnail image for scoring
    rbg_thumbnail  = get_downscaled_thumbnail(slide, to_thumbnail_scale_factor)
    otsu_thumbnail = make_otsu(rbg_thumbnail)

    # get DeepZoomGenerator, level
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)

    tile_x_count, tile_y_count = full_generator.level_tiles[full_level]
    logger.info("tiles x %s, tiles y %s", tile_x_count, tile_y_count)

    # populate address, coordinates
    address_raster = [{"address": coord_to_address(address, requested_magnification), "coordinates": address}
                      for address in itertools.product(range(1, tile_x_count-1), range(1, tile_y_count-1))]
    logger.info("Number of tiles in raster: %s", len(address_raster))

    df = pd.DataFrame(address_raster).set_index("address")

    df.loc[:, "otsu_score"  ] = get_otsu_scores   (df['coordinates'], otsu_thumbnail, thumbnail_tile_size)
    df.loc[:, "purple_score"] = get_purple_scores (df['coordinates'], rbg_thumbnail,  thumbnail_tile_size)

    # get pathology annotations for slide only if valid parameters
    if slideviewer_dmt != None and slideviewer_dmt != "" and labelset != None and labelset != "":
        annotation_url = os.path.join("http://", os.environ['MIND_API_URI'], "mind/api/v1/getPathologyAnnotation/",
                                      slideviewer_dmt, image_id, "regional", labelset)

        response = requests.get(annotation_url)
        response_text = response.text

        if response_text == 'No annotations match the provided query.':
            logger.info("No annotation found for slide.")
        else:
            annotation_geojson = response.json()
            annotation_polygons, annotation_labels = build_shapely_polygons_from_geojson(annotation_geojson)
            df.loc[:, "regional_label"] = get_regional_labels (df['coordinates'], annotation_polygons, annotation_labels, full_generator, full_level)
            
       
    logger.info("Displaying DataFrame for otsu_score > 0.5:")
    logger.info (df [ df["otsu_score"] > 0.5 ])

    output_file = os.path.join(output_dir, "tile_scores_and_labels.csv")

    df.to_csv(output_file)

    logger.info ("Saved tile scores at %s", output_file)

    properties = {
        "data": output_file,
        "magnification": requested_magnification,
        "full_resolution_magnification": requested_magnification * to_mag_scale_factor,
        "tile_size": requested_tile_size,
        "full_resolution_tile_size": full_resolution_tile_size,
        "total_tiles": len(df),
        "available_labels": list(df.columns),
        "tile_magnification": requested_magnification,
        "image_filename": Path(slide_file_path).name
    }

    return properties


### MAIN ENTRY METHOD -> pretile
def run_model(slide_file_path: str, output_dir: str, params: dict):
    """

    Notes: 
    to_mag_scale_factor tells us how much to scale to get from full resolution to desired magnification
    to_thumbnail_scale_factor tells us how much to scale to get from desired magnification to the desired thumbnail downscale, relative to requested mag
    
    The tile size is defined at the requested mag, so it's bigger at full resolution and smaller for the thumbnail
    to_mag_scale_factor and to_thumbnail_scale_factor both need to be event integers, i.e. the scale factors are multiples of the the scanned magnficiation
    """
    logger = logging.getLogger(__name__)

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("magnification")
    model_package             = params.get("model_package")

    logger.info("Processing slide %s", slide_file_path)
    logger.info("Params = %s", params)

    slide = openslide.OpenSlide(str(slide_file_path))

    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])
 
    scale_factor = params.get("scale_factor", 4) # Instead, we should specifiy a thumbnail zoom, and calculate this using get_scale_factor_at_magnfication()
    to_mag_scale_factor         = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)
    to_thumbnail_scale_factor   = to_mag_scale_factor * scale_factor

    if not to_mag_scale_factor % 1 == 0 or not requested_tile_size % scale_factor == 0: 
        raise ValueError("You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size = requested_tile_size * to_mag_scale_factor
    thumbnail_tile_size       = requested_tile_size // scale_factor

    logger.info("Normalized magnification scale factor for %sx is %s, overall thumbnail scale factor is %s", requested_magnification, to_mag_scale_factor, to_thumbnail_scale_factor)
    logger.info("Requested tile size=%s, tile size at full magnficiation=%s, tile size at thumbnail=%s", requested_tile_size, full_resolution_tile_size, thumbnail_tile_size)

    # Create thumbnail image for scoring
    rbg_thumbnail  = get_downscaled_thumbnail(slide, to_thumbnail_scale_factor)
    otsu_thumbnail = make_otsu(rbg_thumbnail)

    # get DeepZoomGenerator, level
    generator, level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)

    tile_x_count, tile_y_count = generator.level_tiles[level]
    logger.info("tiles x %s, tiles y %s", tile_x_count, tile_y_count)

    address_raster = [{"address": coord_to_address(address, requested_magnification), "coordinates": address} for address in itertools.product(range(1, tile_x_count-1), range(1, tile_y_count-1))]
    logger.info("Number of tiles in raster: %s", len(address_raster))

    df_scores = pd.DataFrame(address_raster).set_index("address")

    df_scores.loc[:, "otsu_score"  ] = get_otsu_scores   (df_scores['coordinates'], otsu_thumbnail, thumbnail_tile_size)
    df_scores.loc[:, "purple_score"] = get_purple_scores (df_scores['coordinates'], rbg_thumbnail,  thumbnail_tile_size)
       
    logger.info("Displaying DataFrame for otsu_score > 0.5:")
    logger.info (df_scores [ df_scores["otsu_score"] > 0.5 ])

    logger.info(f"BUILDING MODEL FROM {model_package}..")
    
    tile_model = importlib.import_module(model_package)
    classifier = tile_model.get_classifier ( **params['model'] )
    transform  = tile_model.get_transform ()

    classifier.eval()
    classifier.cuda()

    logger.info( classifier )

    logger.info("RUNNING MODEL...")

    counter = 0
    model_scores = []
    tumor_score  = []
    with torch.no_grad():
        df_tiles_to_process = df_scores[ (df_scores["otsu_score"] > 0.5) &  (df_scores["otsu_score"] > 0.1) ]
        for index, row in df_tiles_to_process.iterrows():
            counter += 1
            if counter % 1000 == 0: logger.info( "Proccessing tiles [%s,%s]", counter, len(df_tiles_to_process))

            output = classifier(transform(generator.get_tile(level, address_to_coord(index)).resize((requested_tile_size,requested_tile_size))).unsqueeze(0).cuda())
            scores = output.exp() / output.exp().sum()

            model_scores.append( 'Label-' + str( scores.argmax(1).item()) )
            tumor_score.append( scores.flatten()[0].item() )

    
    df_tiles_to_process.loc[:, "model_score"] =  model_scores
    df_tiles_to_process.loc[:, "tumor_score"] =  tumor_score

    logger.info(df_tiles_to_process)

    output_file = os.path.join(output_dir, "tile_scores_and_labels_pytorch_inference.csv")
    df_tiles_to_process.to_csv(output_file)

    logger.info ("Saved tile inference data at %s", output_file)

    properties = {
        "data": output_file,
        "magnification": requested_magnification,
        "full_resolution_magnification": requested_magnification * to_mag_scale_factor,
        "tile_size": requested_tile_size,
        "full_resolution_tile_size": full_resolution_tile_size,
        "total_tiles": len(df_tiles_to_process),
        "tile_magnification": requested_magnification,
        "image_filename": Path(slide_file_path).name,
        "available_labels": list(df_tiles_to_process.columns)
    }

    return properties 



### MAIN ENTRY METHOD -> vis tiles
"""
Not used atm
def visualize_scoring(slide_file_path: str, scores_file_path: str, output_dir: str, params: dict):
    logger = logging.getLogger(__name__)

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("magnification")

    logger.info("Processing slide %s", slide_file_path)
    logger.info("Params = %s", params)

    slide = openslide.OpenSlide(str(slide_file_path))

    logger.info("Slide size = [%s,%s]", slide.dimensions[0], slide.dimensions[1])
 
    scale_factor = params.get("scale_factor", 4)
    to_mag_scale_factor         = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)
    to_thumbnail_scale_factor   = to_mag_scale_factor * scale_factor
    
    if not to_mag_scale_factor % 1 == 0 or not requested_tile_size % scale_factor == 0: 
        raise ValueError("You chose a combination of requested tile sizes and magnification that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size   = requested_tile_size * to_mag_scale_factor
    thumbnail_tile_size         = requested_tile_size // scale_factor

    logger.info("Normalized magnification scale factor for %sx is %s, overall thumbnail scale factor is %s", requested_magnification, to_mag_scale_factor, to_thumbnail_scale_factor)
    logger.info("Requested tile size=%s, tile size at full magnficiation=%s, tile size at thumbnail=%s", requested_tile_size, full_resolution_tile_size, thumbnail_tile_size)

    # Create thumbnail image for scoring
    rbg_thumbnail  = get_downscaled_thumbnail(slide, to_thumbnail_scale_factor)
    df_scores      = pd.read_csv(scores_file_path).set_index("address")

    # only visualize tile scores that were able to be computed
    all_score_types = {"tumor_score", "model_score", "purple_score", "otsu_score", "regional_label"}
    score_types_to_visualize = set(list(df_scores.columns)).intersection(all_score_types)

    for score_type_to_visualize in score_types_to_visualize:
        output_file = os.path.join(output_dir, "tile_scores_and_labels_visualization_{}.png".format(score_type_to_visualize))

        thumbnail_overlayed = visualize_tiling_scores(df_scores, rbg_thumbnail, thumbnail_tile_size, score_type_to_visualize)
        thumbnail_overlayed = Image.fromarray(thumbnail_overlayed)
        thumbnail_overlayed.save(output_file)

        logger.info ("Saved %s visualization at %s", score_type_to_visualize, output_file)

    properties = {'data': output_dir}

    return properties
"""

### MAIN ENTRY METHOD -> save tiles
def save_tiles(slide_file_path: str, scores_file_path: str, output_dir: str, params: dict):
    """
    Given slide and tile scores, filter tiles for analysis

    :param slide_file_path: path to svs file
    :param scores_file_path: path to csv file
    :param output_dir: directory to save tiles (.pil) and scores (.csv)
    :param params: job params
    :return: properties with results
    """
    logger = logging.getLogger(__name__)

    logger.info("Processing slide %s", slide_file_path)
    logger.info("Params = %s", params)

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("magnification")

    slide = openslide.OpenSlide(str(slide_file_path))
    df_scores = pd.read_csv(scores_file_path).set_index("address")

    to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=requested_magnification)

    if not to_mag_scale_factor % 1 == 0: 
        raise ValueError("You chose a combination of requested tile sizes and magnifications that resulted in non-integer tile sizes at different scales")

    full_resolution_tile_size = requested_tile_size * to_mag_scale_factor

    generator, level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)

    fp = open(f"{output_dir}/tiles.slice.pil",'wb')
    offset = 0
    counter = 0

    # TODO filter function (column, filter_criteria)
    df_tiles_to_process = df_scores[ (df_scores["otsu_score"] > 0.5) &  (df_scores["otsu_score"] > 0.1) ].dropna()

    for index, row in df_tiles_to_process.iterrows():
        counter += 1
        if counter % 10000 == 0: logger.info( "Proccessing tiles [%s,%s]", counter, len(df_tiles_to_process))

        img_pil     = generator.get_tile(level, address_to_coord(index)).resize((requested_tile_size,requested_tile_size))
        img_bytes   = img_pil.tobytes()

        fp.write( img_bytes )

        df_scores.loc[index, "tile_image_offset"]   = int(offset)
        df_scores.loc[index, "tile_image_length"]   = int(len(img_bytes))
        df_scores.loc[index, "tile_image_size_xy"]  = int(img_pil.size[0])
        df_scores.loc[index, "tile_image_mode"]     = img_pil.mode

        offset += len(img_bytes)
    
    fp.close()

    df_scores.dropna().to_csv(f"{output_dir}/address.slice.csv")

    properties = {
        "data": f"{output_dir}/tiles.slice.pil",
        "aux" : f"{output_dir}/address.slice.csv",
        "tiles": len(df_scores.dropna()),
        "pil_image_bytes_mode": img_pil.mode,
        "pil_image_bytes_size": img_pil.size[0],
        "pil_image_bytes_length": len(img_bytes)
    }

    print (properties)
    return properties

      
