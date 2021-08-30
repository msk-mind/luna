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


def convert_xml_to_mask(xml_fn: str, shape:list, annotation_name:str) -> np.ndarray:
    """convert xml to bitmask
    
    converts a sparse halo XML annotation file (polygons) to a dense bitmask 
    
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
        board_pos = np.zeros(shape, dtype=np.uint8)
        board_neg = np.zeros(shape, dtype=np.uint8)
        regions = ann.findall('Regions')
        assert(len(regions) == 1)
        rs = regions[0].findall('Region')
        plistlist = list()
        nlistlist = list()
        #print('rs:', len(rs))
        for i, r in enumerate(rs):
            ylist = list()
            xlist = list()
            plist, nlist = list(), list()
            negative_flag = int(r.get('NegativeROA'))
            assert negative_flag == 0 or negative_flag == 1
            negative_flag = bool(negative_flag)
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            vs.append(vs[0]) # last dot should be linked to the first dot
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)
                if negative_flag:
                    nlist.append((x, y))
                else:
                    plist.append((x, y))
            if plist:
                plistlist.append(plist)
            else:
                nlistlist.append(nlist)
        for plist in plistlist:
            thiscontour = np.array(plist, dtype=np.int32)
            board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], -1, [255, 0, 0], -1)
        for nlist in nlistlist:
            board_neg = cv2.drawContours(board_neg, [np.array(nlist, dtype=np.int32)], -1, [255, 0, 0], -1)
        ret = (board_pos>0) * (board_neg==0)
    return ret


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
        
    e = et.parse(xml_fn).getroot()
    for ann in e.findall('Annotation'):

        regions = ann.findall('Regions')[0]
        if not regions[0].get('Type')=='Rectangle': continue
        
        for i, r in enumerate(regions):
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)
    return xlist, ylist

def get_slide_roi_masks(slide_path, halo_roi_path, annotation_name, slide_id:str=None,
        output_dir:str=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """get roi masks from slides 

    Given a slide, halo annotation xml file, generate labels from xml polygons, 
    then crop both the image and mask to ROI (rectangle) region
    Optionally: save the RGB image, downsampled image sample, and interger label mask as a tiff 
    
    Args:
        slide_path (str): path to input slide
        halo_roi_path (str): path to halo annotation file
        annotation_name (str): name of annotation 
        slide_id (Optional, str): slide id
        output_dir (Optional, str): destination to save RGB image, thumbnail and mask 
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: the cropped image as RGB numpy array, a 
        downsampled array (as sample for stains), and mask array as single channel
    """

    slide = openslide.OpenSlide(slide_path)
    wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions

    annotation_mask = convert_xml_to_mask(halo_roi_path, wsi_shape, annotation_name)
    x_roi, y_roi    = convert_halo_xml_to_roi(halo_roi_path)

    print (x_roi, y_roi)
#    print ((min(x_roi), min(y_roi)), 0, (abs(x_roi[1] - x_roi[0]), abs(y_roi[1] - y_roi[1])))
    slide_image_cropped  = slide.read_region((min(x_roi), min(y_roi)), 0, (abs(x_roi[1] - x_roi[0]), abs(y_roi[1] - y_roi[0]))).convert('RGB')

    print (slide_image_cropped)

    slide_array  = np.array(slide_image_cropped, dtype=np.uint8)
    sample_array = np.array(slide_image_cropped.resize( (slide_image_cropped.size[0] // 80, slide_image_cropped.size[1] // 80)  ),  dtype=np.uint8)
    mask_array   = annotation_mask[ min(y_roi):max(y_roi), min(x_roi):max(x_roi)].astype(np.uint8)

    if slide_id is not None and output_dir is not None:

        with tifffile.TiffWriter(f'{output_dir}/{slide_id}/{slide_id}_slideImage_roi_inRGB.tiff', bigtiff=True) as tiff:
            tiff.save(slide_array)
        
        with tifffile.TiffWriter(f'{output_dir}/{slide_id}/{slide_id}_slideSample_roi_inRGB.tiff', bigtiff=True) as tiff:
            tiff.save(sample_array)
        
        with tifffile.TiffWriter(f'{output_dir}/{slide_id}/{slide_id}_annotMask_roi_uint8.tiff', bigtiff=True) as tiff:
            tiff.save(mask_array)
    
    return slide_array, sample_array, mask_array


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


def extract_patch_texture_features(address, image_patch, mask_patch, stain_vectors,
        stain_channel, glcm_feature, plot=False) -> np.ndarray:
    """extact patch texture features

    Runs patch-wise extraction from an image_patch, mask_patch pair given a stain
    vector and stain channel.

    Args:
        address (str): unused? 
        image_patch (np.ndarray): input image patch
        mask_patch (np.ndarray): input image mask
        stain_vectors (np.ndarray): stain vectors extacted from the image patch
        stain_channels (int): stain channel 
        glcm_feature (str): unused? 
        plot (Optional, bool): unused? 

    Returns:
        np.ndarray: texture features from image patch
    
    """

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binWidth=16)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(glcm=['ClusterTendency'])
    extractor.enableImageTypeByName('Original')
    
    logger.debug (f"Label sum= {mask_patch.sum()}")
   
    #mask_patch = np.array( Image.fromarray(mask_patch).resize((250,250))).astype(np.uint8)
    if not (len(np.unique(mask_patch)) > 1 and np.count_nonzero(mask_patch) > 1): return None
    
    stain_patch = pull_stain_channel(image_patch, stain_vectors, channel=stain_channel)
    
    #stain_patch = np.array( Image.fromarray(stain_patch).resize((250,250)))
    
    sitk_image  = sitk.GetImageFromArray(stain_patch.astype(np.uint8))
    sitk_mask   = sitk.GetImageFromArray(mask_patch.astype(np.uint8))

    try:
        bbox, _ = radiomics.imageoperations.checkMask(sitk_image, sitk_mask)
    except Exception as exc:
        logger.warning (f"Skipping this patch, mask pair due to '{exc}'")
    else:
        cimg, cmas = radiomics.imageoperations.cropToTumorMask(sitk_image, sitk_mask, bbox)

        fts = extractor.execute(sitk_image, sitk_mask, voxelBased=True)

        cluster_tend_patch   = sitk.GetArrayFromImage(fts['original_glcm_ClusterTendency']).astype(np.float).flatten()
        cluster_tend_nonzero = cluster_tend_patch[cluster_tend_patch != 0]
        cluster_tend_valid   = cluster_tend_nonzero[~np.isnan(cluster_tend_nonzero)]

        return cluster_tend_valid
            
