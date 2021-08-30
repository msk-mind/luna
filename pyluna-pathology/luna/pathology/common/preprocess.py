"""
@author: aukermaa@mskcc.org
@author: pateld6@mskcc.org
@author: rosed2@mskcc.org

Various utility and processing methods for pathology
"""

import os, itertools, logging, re

from typing import Union, Tuple, List, Dict
import numpy  as np
import pandas as pd
import seaborn as sns

import json 
from pathlib import Path

from PIL import Image
from pyarrow.parquet import read_table

import openslide
from openslide.deepzoom import DeepZoomGenerator

from skimage.color   import rgb2gray
from skimage.filters import threshold_otsu
from skimage.draw import rectangle_perimeter, rectangle

import requests
import importlib

from shapely.geometry import shape, Point, Polygon

from luna.common.DataStore import DataStore_v2
from random import randint
import torch

logger = logging.getLogger(__name__)


palette = sns.color_palette("viridis",as_cmap=True)
categorial = sns.color_palette("Set1", 8)
categorical_colors = {}

def get_tile_color(score:Union[str, float]) -> Union[float, None]:
    """get tile color

    uses deafult color palette to return color of tile based on score

    Args:
        score (Union[str, float]): a value between [0,1] such as the 
            Otsu threshold, puple score, a model output, etc. 
    Returns:
        Union[float, None]: returns the color is the input is of valid type
            else None

    """
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
def array_to_slide(arr: np.ndarray) -> openslide.OpenSlide:
    """converts a numpy array to a openslide.OpenSlide object

    Args:
        arr (np.ndarray): input image array
    
    Returns:
        openslide.OpenSlide: a slide object from openslide    
    """

    assert isinstance(arr, np.ndarray)
    slide = openslide.ImageSlide(Image.fromarray(arr))
    return slide
   

# USED -> utils
def get_scale_factor_at_magnfication(slide: openslide.OpenSlide,
        requested_magnification: int) -> int:
    """get scale factor at magnification

    Return a scale factor if slide scanned magnification and
    requested magnification are different.

    Args:
        slide (openslide.OpenSlide): slide object
        requested_magnification (int): requested magnification
    
    Returns:
        int: scale factor required to achieve requested magnification
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
            raise ValueError(f'Expected magnification <={scanned_magnfication} but got {requested_magnification}')
        elif (scanned_magnfication % requested_magnification) == 0:
            scale_factor = scanned_magnfication // requested_magnification
        else:
            raise ValueError(f'Expected magnification {requested_magnification} to be an divisor multiple of {scanned_magnfication}')
    return scale_factor


# USED -> utils
def get_full_resolution_generator(slide: openslide.OpenSlide, tile_size:int) -> Tuple[DeepZoomGenerator, int]:
    """Return DeepZoomGenerator and generator level

    Args:
        slide (openslide.OpenSlide): slide object
        tile_size (int): width and height of a single tile for the DeepZoomGenerator
    
    Returns:
        Tuple[DeepZoomGenerator, int] 
    """
    assert isinstance(slide, openslide.OpenSlide) or isinstance(slide, openslide.ImageSlide)
    generator = DeepZoomGenerator(slide, overlap=0, tile_size=tile_size, limit_bounds=False)
    generator_level = generator.level_count - 1
    assert generator.level_dimensions[generator_level] == slide.dimensions
    return generator, generator_level

# USED -> generate cli
def get_otsu_scores(address_raster:list, otsu_img:np.ndarray, otsu_tile_size:int) -> List[float]:
    """compute otsu score

    otsu thresholds a single tile and computes the mean value to yield a scalar score
    between 0 and 1

    Args:
        addresss_raster (list): the raster address of a tiles to score
        otsu_img (np.ndarray): input array to threshold
        otsu_tile_size (int): size of input tiles  
    
    Returns:
        list[float]: a list of otsu scores 
    """
    otsu_slide = array_to_slide(otsu_img)
    otsu_generator, otsu_generator_level = get_full_resolution_generator(otsu_slide, tile_size=otsu_tile_size)
    otsu_score_results = []
    for address in address_raster:
        otsu_tile = np.array(otsu_generator.get_tile(otsu_generator_level, address))
        otsu_score_results.append( otsu_tile.mean().item() )
    return otsu_score_results

# USED -> generate cli
def get_purple_scores(address_raster:list, rgb_img:np.ndarray, rgb_tile_size:int) -> List[float]:
    """compute purple score 

    computes the purple score for tiles in a slide 

    Args:
        address_raster (list): the raster address of tiles to score
        rgb_img (np.ndarray): input array to score
        rgb_tile_size (int): size of input tiles 
    
    Returns:
        list[float]: a list of purple scores    
    """

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
        purple_score_results.append(score)  

    return purple_score_results

# USED -> utils
def coord_to_address(s:Tuple[int, int], magnification:int)->str:
    """converts coordinate to address

    Args:
        s (tuple[int, int]): coordinate consisting of an (x, y) tuple
        magnification (int): magnification factor

    Returns:
        str: a string consisting of an x_y_z address
    """

    x = s[0]
    y = s[1]
    return f"x{x}_y{y}_z{magnification}"

# USED -> utils
def address_to_coord(s:str) -> Tuple[int, int]:
    """converts address into coordinates
    
    Args:
        s (str): a string consisting of an x_y_z address 

    Returns:
        Tuple[int, int]: a tuple consisting of an x, y pair 
    """
    s = str(s)
    p = re.compile('x(\d+)_y(\d+)_z(\d+)', re.IGNORECASE)
    m = p.match(s)
    x = int(m.group(1))
    y = int(m.group(2))
    return (x,y)

# USED -> utils
def get_downscaled_thumbnail(slide:openslide.OpenSlide, scale_factor:int)-> np.ndarray:
    """get downscaled thumbnail

    yields a thumbnail image of a whole slide rescaled by a specified scale factor

    Args:
        slide (openslide.OpenSlide): slide object
        scale_factor (int): integer scaling factor to resize the whole slide by
    
    Returns:    
        np.ndarray: downsized whole slie thumbnail 
    """
    new_width  = slide.dimensions[0] // scale_factor
    new_height = slide.dimensions[1] // scale_factor
    img = slide.get_thumbnail((new_width, new_height))
    return np.array(img)

# USED -> generate tiles
def make_otsu(img: np.ndarray, scale:int=1) -> np.ndarray: 
    """make otsu-thresholded image 
    
    Make image with pixel-wise foreground/background labels as determined by the otsu
    threshold 

    Args:
        img (np.ndarray):  grayscale input image
        scale (float): value to scale the threshold by 

    Returns:
        np.ndarray: array where each pixel is 0 if background and 1 if foreground
    """
    assert isinstance(img, np.ndarray)
    _img = rgb2gray(img)
    threshold = threshold_otsu(_img)
    return (_img < (threshold * scale)).astype(float)

def build_shapely_polygons_from_geojson(annotation_geojson:Dict[str, any])-> Tuple[list,
        list]:
    """Build shapely polygons from geojson

    builds a list of shapely polygons and their cooresponding label from a geojson object

    Args:
        annotation_geojson (dict[str, any]): input annotation geoJSON object
    
    Returns:
        Tuple[list, list]: a tuple consisting of polygon and annotation label lists
    """
    annotation_polygons = []
    annotation_labels = []
    # print(len(annotation_geojson['features']))

    for feature in annotation_geojson['features']:

        class_name = feature['properties']['label_name']

        ring_list = feature['geometry']['coordinates']

        # polygon with no holes
        if len(ring_list) == 1:
            annotation_polygon = Polygon(ring_list[0])
        else:
            # this is a ring with interior holes
            annotation_polygon = Polygon(ring_list[0], holes=ring_list[1:])
        
        # check all coord lists are valid

        if annotation_polygon.is_valid:
            annotation_polygons.append(annotation_polygon)
            annotation_labels.append(class_name)


    return annotation_polygons,annotation_labels


def get_regional_labels(address_raster:list, annotation_polygons:list,
        annotation_labels:list, full_generator:DeepZoomGenerator, full_level:int)->list:
    """get regional labels

    Returns annotation labels for tiles that contain annotations
    If the tile doesn't contain annotation, set label to None.

    Args:
        address_raster (list): raster coordinates for tiles 
        annotation_polygons (list): list of shapely Polygon objects
        annotation_labels (list): list of annotation label names
        full_generator (DeepZoomGenerator): whole slide full resolution generator
        param full_level (int): full res level for full_generator
    
    Returns:
        list: list of annotation labels for each polygon 
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
def pretile_scoring(slide_file_path: str, output_dir: str, annotation_table_path: str,
        params: dict, image_id: str) -> dict:
    """preform tiling operations and score tiles 

    Generate tiles and scores. to_mag_scale_factor tells us how much to scale to get from
    full resolution to desired magnification. to_thumbnail_scale_factor tells us how much to 
    scale to get from desired magnification to the desired thumbnail downscale, relative
    to the requested mag. The tile size is defined at the requested mag, so it's bigger at 
    full resolution and smaller for the thumbnail to_mag_scale_factor and to_thumbnail_scale_factor
    both need to be event integers, i.e. the scale factors are multiples 
    of the the scanned magnficiation

    Args:
        slide_file_path (str): input whole slide file path
        output_dir (str): directory to save files
        annotation_table_path (str): path to annotation table
        params (dict): parameter dict consisting of tile_size, magnification,
            project_id, label_set, filter and scale factor
        image_id (str): input image id 

    Returns:
        dict: a dictionary of properties specifying parameters used to generate tiles
            and save output files
    """

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("requested_magnification")

    # optional arguments related to slideviewer annotations
    project_id               = params.get("project_id", None)
    labelset                  = params.get("labelset", None)
    filter                    = params.get("filter")

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

    # TODO custom scoring
    df.loc[:, "otsu_score"  ] = get_otsu_scores   (df['coordinates'], otsu_thumbnail, thumbnail_tile_size)
    df.loc[:, "purple_score"] = get_purple_scores (df['coordinates'], rbg_thumbnail,  thumbnail_tile_size)

    # get pathology annotations for slide only if valid parameters
    if project_id != None and project_id != "" and labelset != None and labelset != "":
        # from get_pathology_annotations
        regional_annotation_table = read_table(annotation_table_path, columns=["geojson_path"],
                                               filters=[('slide_id','=',f'{image_id}'),
                                                        ('user', '=', 'CONCAT'),
                                                        ('labelset', '=', f'{labelset.upper()}')]) \
                                    .to_pandas()
        geojson_path = regional_annotation_table['geojson_path'][0]
        if geojson_path:
            # geojson is saved as a string
            with open(geojson_path) as geojson_file:
                annotation_geojson = json.loads(json.load(geojson_file))

            annotation_polygons, annotation_labels = build_shapely_polygons_from_geojson(annotation_geojson)
            df.loc[:, "regional_label"] = get_regional_labels (df['coordinates'], annotation_polygons, annotation_labels, full_generator, full_level)

    fp = open(f"{output_dir}/tiles.slice.pil",'wb')
    offset = 0
    counter = 0

    # filter tiles based on user provided criteria
    # This line isn't neccessary if we don't do inplace operations on df_tiles_to_process
    df_tiles_to_process = df
   
    if filter is not None:
        for column, threshold in filter.items():
            df_tiles_to_process = df_tiles_to_process[df_tiles_to_process[column] >= threshold]

    for index, row in df_tiles_to_process.iterrows():
        counter += 1
        if counter % 10000 == 0: logger.info( "Proccessing tiles [%s,%s]", counter, len(df_tiles_to_process))

        img_pil     = full_generator.get_tile(full_level, address_to_coord(index)).resize((requested_tile_size,requested_tile_size))
        img_bytes   = img_pil.tobytes()

        fp.write( img_bytes )

        df_tiles_to_process.loc[index, "tile_image_offset"]   = int(offset)
        df_tiles_to_process.loc[index, "tile_image_length"]   = int(len(img_bytes))
        df_tiles_to_process.loc[index, "tile_image_size_xy"]  = int(img_pil.size[0])
        df_tiles_to_process.loc[index, "tile_image_mode"]     = img_pil.mode

        offset += len(img_bytes)

    fp.close()

    # drop null columns
    df_tiles_to_process.dropna() \
        .to_csv(f"{output_dir}/address.slice.csv")

    properties = {
        "data": f"{output_dir}/tiles.slice.pil",
        "aux" : f"{output_dir}/address.slice.csv",
        "tiles": len(df_tiles_to_process.dropna()),
        "tile_magnification": requested_magnification,
        "full_resolution_magnification": requested_magnification * to_mag_scale_factor,
        "tile_size": requested_tile_size,
        "full_resolution_tile_size": full_resolution_tile_size,
        "image_filename": Path(slide_file_path).name,
        "available_labels": list(df.columns),
        "pil_image_bytes_mode": img_pil.mode,
        "pil_image_bytes_size": img_pil.size[0],
        "pil_image_bytes_length": len(img_bytes)
    }

    logger.info ("Saved tile scores and images at %s", output_dir)

    return properties


### MAIN ENTRY METHOD -> pretile
def run_model(pil_file_path: str, csv_file_path: str, output_dir: str, params: dict) -> dict:
    """runs a tile classifier model on a tile data frame/csv
    
    Loads a PyTorch model and runs inference on a set of tiles in an input dataframe. 
    The results are saved to an output csv. 

    to_mag_scale_factor tells us how much to scale to get from full resolution to
    desired magnification. to_thumbnail_scale_factor tells us how much to scale to get
    from desired magnification to the desired thumbnail downscale, relative to requested
    mag. The tile size is defined at the requested mag, so it's bigger at full resolution
    and smaller for the thumbnail. to_mag_scale_factor and to_thumbnail_scale_factor both
    need to be event integers, i.e. the scale factors are multiples of the the scanned
    magnficiation

    Args:
        pil_file_path (str): file path to the input whole slide 
        csv_file_path (str): input tile csv/dataframe 
        output_dir (str): destination to save inference results to 
        params (dict): configuration dictionary consisting of model_package, which
            properties of the tile classifier model 

    Returns:
        properties (dict): a properties dictionary with return values 
        
    """ 
    model_package             = params.get("model_package")

    # load csv
    df_tiles_to_process = pd.read_csv(csv_file_path)

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
    label_score = []

    fp = open(pil_file_path, "rb")

    with torch.no_grad():

        for index, row in df_tiles_to_process.iterrows():
            counter += 1
            if counter % 1000 == 0: logger.info( "Proccessing tiles [%s,%s]", counter, len(df_tiles_to_process))

            # TODO load tiles from pil
            fp.seek(int (row.tile_image_offset))
            img = Image.frombytes(row.tile_image_mode,
                                  (int (row.tile_image_size_xy), int (row.tile_image_size_xy)),
                                  fp.read(int (row.tile_image_length)))

            output = classifier(transform(img).unsqueeze(0).cuda())
            scores = output.exp() / output.exp().sum()

            model_scores.append( 'Label-' + str( scores.argmax(1).item()) )
            tumor_score.append( scores.flatten()[0].item() )
            label_score.append( scores.max().item() )
            
    
    df_tiles_to_process.loc[:, "model_score"] =  model_scores
    df_tiles_to_process.loc[:, "tumor_score"] =  tumor_score
    df_tiles_to_process.loc[:, "label_score"] =  label_score

    logger.info(df_tiles_to_process)

    output_file = os.path.join(output_dir, "tile_scores_and_labels_pytorch_inference.csv")
    df_tiles_to_process.to_csv(output_file)

    logger.info ("Saved tile inference data at %s", output_file)

    properties = {
        "data": output_file,
        "total_tiles": len(df_tiles_to_process),
        "image_filename": Path(pil_file_path).name,
        "available_labels": list(df_tiles_to_process.columns)
    }

    return properties 

def create_tile_thumbnail_image(slide_file_path: str, scores_file_path: str, output_dir:
        str, params: dict)-> dict:
    """creates thumbnail images for score visualizations
    
    creates a thumbnail image for score visualizations (otsu, purple, model scores, etc.) and
    saves the result to an output directory. 

    Args:
        slide_file_path (str): file path of slide to visualize
        scores_file_path (str): file path of score .csv file to visualize
        output_dir (str): destination to save thumbnail image to 
        params (dict): parmater dictionary consisting of tile_size, magnification,
            and scale_factor

    Returns:
        dict: a properties dictionary with return values 
    """

    requested_tile_size       = params.get("tile_size")
    requested_magnification   = params.get("requested_magnification")

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
    all_score_types = {"tumor_score", "model_score", "label_score", "purple_score", "otsu_score", "regional_label"}
    score_types_to_visualize = set(list(df_scores.columns)).intersection(all_score_types)

    for score_type_to_visualize in score_types_to_visualize:
        output_file = os.path.join(output_dir, "tile_scores_and_labels_visualization_{}.png".format(score_type_to_visualize))

        thumbnail_overlayed = visualize_tiling_scores(df_scores, rbg_thumbnail, thumbnail_tile_size, score_type_to_visualize)
        thumbnail_overlayed = Image.fromarray(thumbnail_overlayed)
        thumbnail_overlayed.save(output_file)

        logger.info ("Saved %s visualization at %s", score_type_to_visualize, output_file)

    properties = {'data': output_dir}

    return properties


def visualize_tiling_scores(df:pd.DataFrame, thumbnail_img:np.ndarray,
        tile_size:int, score_type_to_visualize:str) -> np.ndarray:
    """visualize tile scores
    
    draws colored boxes around tiles to indicate the value of the score 

    Args:
        df (pd.DataFrame): input dataframe
        thumbnail_img (np.ndarray): input tile 
        tile_size (int): tile width/length
        score_type_to_visualize (str): column name from data frame
    
    Returns:
        np.ndarray: new thumbnail image with boxes around tiles passing indicating the
        value of the score
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

