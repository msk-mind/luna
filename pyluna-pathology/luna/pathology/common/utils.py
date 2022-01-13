### This is just luna.pathology/common/utils.py

from   luna.common.config import ConfigSet
import luna.common.constants as const

from typing import Tuple, List
import xml.etree.ElementTree as et
import numpy as np
import cv2

import radiomics
import SimpleITK as sitk

import openslide
import tifffile

import logging

from matplotlib import pyplot as plt
from PIL import Image

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


def pull_stain_channel(patch:np.ndarray, vectors:np.ndarray, channel:int=0)->np.ndarray:
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
    return tmp[:,:,channel]


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
