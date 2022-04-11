### This is just luna.pathology/common/utils.py

from   luna.common.config import ConfigSet
import luna.common.constants as const

from typing import Union, Tuple, List
import xml.etree.ElementTree as et
import numpy as np
import cv2

import radiomics
import SimpleITK as sitk
import re
import h5py

import openslide

import logging

from PIL import Image

from openslide.deepzoom import DeepZoomGenerator
import seaborn as sns

from skimage.draw import rectangle_perimeter
from tqdm import tqdm
import pandas as pd

from skimage.color   import rgb2gray, rgba2rgb

palette = sns.color_palette("viridis",as_cmap=True)
categorial = sns.color_palette("Set1", 8)
categorical_colors = {}

logger = logging.getLogger(__name__)

def get_labelset_keys():
    """get labelset keys
    
    Given DATA_CFG, return slideviewer labelsets
    
    Args:
        none
    
    Returns:
        list: a list of labelset names 
    """
    cfg = ConfigSet()
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    labelsets = [cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]

    if cfg.get_value(path=const.DATA_CFG+'::USE_ALL_LABELSETS'):
        labelsets = list(label_config.keys())

    return labelsets

def get_layer_names(xml_fn):
    """get available layer names
    
    Finds all possible annotation layer names from a Halo generated xml ROI file 
    
    Args:
        xml_fn (str): file path to input halo XML file
    
    Returns:
        set: Available region names
    """    # Annotations >> 
    e = et.parse(xml_fn).getroot()
    e = e.findall('Annotation')
    names = set()        

    [names.add(ann.get('Name')) for ann in e]

    return names
    

def convert_xml_to_mask(xml_fn: str, shape:list, annotation_name:str) -> np.ndarray:
    """convert xml to bitmask
    
    Converts a sparse halo XML annotation file (polygons) to a dense bitmask 
    
    Args:
        xml_fn (str): file path to input halo XML file
        shape (list): desired polygon shape
        annotation_name (str): name of annotation 
    
    Returns:
        np.ndarray: annotation bitmask of specified shape
    """

    ret = None
    board_pos = None
    board_neg = None
    # Annotations >> 
    e = et.parse(xml_fn).getroot()
    e = e.findall('Annotation')
    for ann in e:
        if ann.get('Name') != annotation_name:
                continue

        logger.debug(f"Found region {ann.get('Name')}")

        board_pos = np.zeros(shape, dtype=np.uint8)
        board_neg = np.zeros(shape, dtype=np.uint8)

        regions = ann.findall('Regions')
        assert(len(regions) == 1)

        rs = regions[0].findall('Region')
        
        for i, r in enumerate(rs):

            negative_flag = int(r.get('NegativeROA'))
            assert negative_flag == 0 or negative_flag == 1
            negative_flag = bool(negative_flag)
            
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            vs.append(vs[0]) # last dot should be linked to the first dot

            plist = list()
            for v in vs:
                x, y = int(v.get('X').split('.')[0]), int(v.get('Y').split('.')[0])
                plist.append((x, y))

            if negative_flag:
                board_neg = cv2.drawContours(board_neg, [np.array(plist, dtype=np.int32)], -1, [0, 0, 0], -1)
            else:
                board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], contourIdx=-1, color=[255, 0, 0], thickness=-1)

        ret = (board_pos>0) * (board_neg==0)

    mask = ret.astype(np.uint8)
    return mask


def convert_halo_xml_to_roi(xml_fn:str) -> Tuple[List, List]:
    """get roi from halo XML file

    Read the rectangle ROI of a halo XML annotation file 
    
    Args:
        xml_fn: file path to input halo XML file 

    Returns:
        Tuple[list, list]: returns a tuple of x, y coordinates of the recangular roi

    """
    
    ylist = list()
    xlist = list()

    print ("Converting to ROI:", xml_fn) 
    e = et.parse(xml_fn).getroot()
    for ann in e.findall('Annotation'):

        regions = ann.findall('Regions')[0]
        if len(regions)==0: continue

        if not regions[0].get('Type')=='Rectangle': continue
        
        for i, r in enumerate(regions):
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)

    if xlist == [] or ylist == []:
        logger.warning ("No Rectangle found, returning None, None!")
        return None, None

    if min(xlist) < 0: 
        logger.warning ("Somehow a negative x rectangle coordinate!")
        xlist = [0, max(xlist)] 
    if min(ylist) < 0: 
        logger.warning ("Somehow a negative y rectangle coordinate!")
        ylist = [0, max(ylist)] 

    return xlist, ylist


def get_stain_vectors_macenko(sample: np.ndarray):
    """get_stain_vectors
    
    Uses the staintools MacenkoStainExtractor to extract stain vectors 
    
    Args:
        sample (np.ndarray): input patch
    Returns:
        np.ndarray: the stain matrix 
    
    """
    from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor

    extractor = MacenkoStainExtractor()
    vectors = extractor.get_stain_matrix(sample)
    return vectors


def pull_stain_channel(patch:np.ndarray, vectors:np.ndarray, channel:int=None)->np.ndarray:
    """pull stain channel
    
    adds 'stain channel' to the image patch

    Args:
        patch (np.ndarray): input image patch
        vectors (np.ndarray): stain vectors
        channel (int): stain channel
    
    Returns:
        np.ndarray: the input image patch with an added stain channel
    """

    from staintools.miscellaneous.get_concentrations import get_concentrations

    tile_concentrations = get_concentrations(patch, vectors)
    identity = np.array([[1,0,0],[0,1,0]])
    tmp = 255 * (1 - np.exp(-1 * np.dot(tile_concentrations, identity)))
    tmp = tmp.reshape(patch.shape).astype(np.uint8)   
    if not channel is None:
        return tmp[:,:,channel]
    else:
        return tmp


def extract_patch_texture_features(image_patch, mask_patch, stain_vectors,
        stain_channel, glcm_feature, plot=False) -> np.ndarray:
    """extact patch texture features

    Runs patch-wise extraction from an image_patch, mask_patch pair given a stain
    vector and stain channel.

    Args:
        image_patch (np.ndarray): input image patch
        mask_patch (np.ndarray): input image mask
        stain_vectors (np.ndarray): stain vectors extacted from the image patch
        stain_channel (int): stain channel 
        glcm_feature (str): unused? 
        plot (Optional, bool): unused? 

    Returns:
        np.ndarray: texture features from image patch
    
    """

    logging.getLogger('radiomics.featureextractor').setLevel(logging.WARNING)
    if not (len(np.unique(mask_patch)) > 1 and np.count_nonzero(mask_patch) > 1): return None
    
    stain_patch = pull_stain_channel(image_patch, stain_vectors, channel=stain_channel)

    if glcm_feature==None:
        original_pixels = stain_patch.astype(np.uint8)[np.where(mask_patch.astype(np.bool))].flatten()
        original_pixels_valid = original_pixels[original_pixels > 0]
        return original_pixels_valid 

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binWidth=16)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(glcm=[glcm_feature])
    extractor.enableImageTypeByName('Original')

    sitk_image  = sitk.GetImageFromArray(stain_patch.astype(np.uint8))
    sitk_mask   = sitk.GetImageFromArray(mask_patch. astype(np.uint8))

    try:
        bbox, _ = radiomics.imageoperations.checkMask(sitk_image, sitk_mask)
    except Exception as exc:
        logger.warning (f"Skipping this patch, mask pair due to '{exc}'")
        return None
    else:
        # cimg, cmas = radiomics.imageoperations.cropToTumorMask(sitk_image, sitk_mask, bbox)

        fts = extractor.execute(sitk_image, sitk_mask, voxelBased=True)

        stainomics_patch   = sitk.GetArrayFromImage(fts[f'original_glcm_{glcm_feature}']).astype(np.float)
        stainomics_nonzero = stainomics_patch[stainomics_patch != 0].flatten()
        stainomics_valid   = stainomics_nonzero[~np.isnan(stainomics_nonzero)]

        return stainomics_valid


def get_tile_from_slide(tile_row, slide, size=None):
    x, y, extent = int (tile_row.x_coord), int (tile_row.y_coord), int (tile_row.xy_extent)
    if size is None:
        size = (tile_row.tile_size, tile_row.tile_size)
    tile = np.array(slide.read_region((x, y), 0, (extent, extent)).resize(size, Image.NEAREST))[:, :, :3]
    return tile


def get_tile_arrays(indices: List[int], input_slide_image: str, full_resolution_tile_size: int, tile_size: int) -> np.ndarray:
    """
    Get tile arrays for the tile indices

    Args:
        indices (List[int]): list of integers to return as tiles
        input_slide_image (str): path to WSI
        full_resolution_tile_size (int): tile_size * to_mag_scale_factor
        tile_size (int): width, height of generated tile

    Returns:
        a list of tuples (index, tile array) for given indices
    """
    slide = openslide.OpenSlide(str(input_slide_image))
    full_generator, full_level = get_full_resolution_generator(slide, tile_size=full_resolution_tile_size)
    return [(index, np.array(full_generator.get_tile(full_level, address_to_coord(index)).resize((tile_size,tile_size))))
            for index in indices]

def get_tile_array(row: pd.DataFrame) -> np.ndarray:
    """
    Returns a tile image as a numpy array.

    Args:
        row (pd.DataFrame): row with address and tile_image_file columns
    """
    with h5py.File(row.tile_store, 'r') as hf:
        tile = np.array(hf[row.name])
    return tile

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

    # # Make sure we don't have non-integer magnifications
    if not int (mag_value) == mag_value:
        raise RuntimeError("Can't handle slides scanned at non-integer magnficiations! (yet)")

    # Verify magnification valid
    scale_factor = 1
    if scanned_magnfication != requested_magnification:
        if scanned_magnfication < requested_magnification:
            raise ValueError(f'Expected magnification <={scanned_magnfication} but got {requested_magnification}')
        elif (scanned_magnfication % requested_magnification) == 0:
            scale_factor = scanned_magnfication // requested_magnification
        else:
            logger.warning("Scale factor is not an integer, be careful!")
            scale_factor = scanned_magnfication / requested_magnification
            
    return scale_factor

def visualize_tiling_scores(df:pd.DataFrame, thumbnail_img:np.ndarray, scale_factor:float,
        score_type_to_visualize:str, normalize=True) -> np.ndarray:
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

    assert isinstance(thumbnail_img, np.ndarray)

    if normalize and df[score_type_to_visualize].dtype.kind in 'biuf':
        df[score_type_to_visualize] = (df[score_type_to_visualize] - np.min(df[score_type_to_visualize]))/np.ptp(df[score_type_to_visualize])

    for _, row in tqdm(df.iterrows(), total=len(df)):

        if 'regional_label' in row and pd.isna(row.regional_label): continue

        start = (row.y_coord / scale_factor, row.x_coord / scale_factor)  # flip because OpenSlide uses (column, row), but skimage, uses (row, column)

        rr, cc = rectangle_perimeter(start=start, extent=(row.xy_extent/ scale_factor, row.xy_extent/ scale_factor), shape=thumbnail_img.shape)
        
        # set color based on intensity of value instead of black border (1)
        score = row[score_type_to_visualize]

        thumbnail_img[rr, cc] = get_tile_color(score)
    
    return thumbnail_img

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