### This is just data_processing/pathology/common/utils.py

from   data_processing.common.config import ConfigSet
import data_processing.common.constants as const

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
    """
    Given DATA_CFG, return slideviewer labelsets

    :return: list of labelset names
    """
    cfg = ConfigSet()
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    labelsets = [cfg.get_value(path=const.DATA_CFG+'::USE_LABELSET')]

    if cfg.get_value(path=const.DATA_CFG+'::USE_ALL_LABELSETS'):
        labelsets = list(label_config.keys())

    return labelsets


def get_polygon_bounding_box(xml_fn, annotation_name):
    """ Convert a sparse XML annotation file (polygons) to a dense bitmask of shape <shape> """

    # Annotations >> 
    e = et.parse(xml_fn).getroot()
    e = e.findall('Annotation')
    ylist = list()
    xlist = list()
    for ann in e:
        if ann.get('Name') != annotation_name:
                continue
        regions = ann.findall('Regions')
        assert(len(regions) == 1)
        rs = regions[0].findall('Region')
        #print('rs:', len(rs))
        for i, r in enumerate(rs):
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)
    print ([min(xlist), max(xlist)], [min(ylist), max(ylist)])

    return [min(xlist), max(xlist)], [min(ylist), max(ylist)] 

def convert_xml_to_mask(xml_fn, shape, annotation_name):
    """ Convert a sparse XML annotation file (polygons) to a dense bitmask of shape <shape> """

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
            board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], -1, [255, 0, 0], -1)
        for nlist in nlistlist:
            board_neg = cv2.drawContours(board_neg, [np.array(nlist, dtype=np.int32)], -1, [255, 0, 0], -1)
        ret = (board_pos>0) * (board_neg==0)
    return ret


def convert_halo_xml_to_roi(xml_fn):
    """ Read the rectangle ROI of a halo XML annotation file """
    
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

def get_slide_roi_masks(slide_path, halo_roi_path, annotation_name, slide_id=None, output_dir=None, scale_factor=None):
    """ 
    Given a slide, halo annotation xml file, generate labels from xml polygons, then, crop both the image and mask to ROI (rectangle) region
    Optionally: save the RGB image, downsampled image sample, and interger label mask as a tiff 
    returns: slide_array, sample_array, mask_array tuple: the cropped image as RGB numpy array, a downsampled array (as sample for stains), and mask array as single channel
    """

    slide = openslide.OpenSlide(slide_path)
    wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions

    x_pol, y_pol    = get_polygon_bounding_box(halo_roi_path, annotation_name)
    x_roi, y_roi    = convert_halo_xml_to_roi(halo_roi_path)

    annotation_mask = convert_xml_to_mask(halo_roi_path, wsi_shape, annotation_name)

    if x_roi == None:
        x_roi = [0, wsi_shape[0]]
        y_roi = [0, wsi_shape[1]]
        logger.warning ("No Rectangle ROI detected, using full slide!!!")

    print ("ROI bounds:", x_pol, y_pol, x_roi, y_roi)
    x_bound = [ max ( [min(x_pol), min(x_roi)] ), min ( [max(x_pol), max(x_roi)] ) ]
    y_bound = [ max ( [min(y_pol), min(y_roi)] ), min ( [max(y_pol), max(y_roi)] ) ]

    logger.info (f"Final bounding box= x, y: {x_bound}, {y_bound}")

    slide_image_cropped  = slide.read_region((x_bound[0], y_bound[0]), 0, (x_bound[1] - x_bound[0], y_bound[1] - y_bound[0] )).convert('RGB')
    mask_image_cropped   = Image.fromarray(annotation_mask[ y_bound[0]:y_bound[1], x_bound[0]:x_bound[1] ].astype(np.uint8))
    sample_image_cropped = slide_image_cropped.resize ( (slide_image_cropped.width // 20, slide_image_cropped.height // 20) )
    print (slide_image_cropped, mask_image_cropped)

    if scale_factor is not None:
        slide_image_cropped = slide_image_cropped.resize( (slide_image_cropped.width // scale_factor,  slide_image_cropped.height // scale_factor))
        mask_image_cropped  = mask_image_cropped.resize ( (mask_image_cropped.width  // scale_factor,  mask_image_cropped.height  // scale_factor))

    if output_dir is not None:
        slide_image_cropped.resize( (slide_image_cropped.width // 5,  slide_image_cropped.height // 5)).save(f'{output_dir}/slide_image_out.png')
        Image.fromarray( 255 * np.array(mask_image_cropped)).resize ( (mask_image_cropped.width  // 5,  mask_image_cropped.height  // 5)).save(f'{output_dir}/mask_image_out.png')
        sample_image_cropped.save(f'{output_dir}/sample_image_out.png')

    slide_array  = np.array(slide_image_cropped,  dtype=np.uint8)
    sample_array = np.array(sample_image_cropped, dtype=np.uint8)
    mask_array   = np.array(mask_image_cropped,   dtype=np.uint8)

  
    return slide_array, sample_array, mask_array


def get_stain_vectors_macenko(sample):
    """ Use the staintools MacenkoStainExtractor to extract stain vectors """
    from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor

    extractor = MacenkoStainExtractor()
    vectors = extractor.get_stain_matrix(sample)
    return vectors


def pull_stain_channel(patch, vectors, channel=0):
    from staintools.miscellaneous.get_concentrations import get_concentrations

    tile_concentrations = get_concentrations(patch, vectors)
    identity = np.array([[1,0,0],[0,1,0]])
    tmp = 255 * (1 - np.exp(-1 * np.dot(tile_concentrations, identity)))
    tmp = tmp.reshape(patch.shape).astype(np.uint8)   
    return tmp[:,:,channel]


def extract_patch_texture_features(address, image_patch, mask_patch, stain_vectors, stain_channel, glcm_feature, plot=False):
    """
    Runs patch-wise extraction from an image_patch, mask_patch pair given a stain vector and stain channel.
    """

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binWidth=16)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(glcm=[glcm_feature])
    extractor.enableImageTypeByName('Original')
    
    logger.debug (f"Label sum= {mask_patch.sum()}")
   
    if not (len(np.unique(mask_patch)) > 1 and np.count_nonzero(mask_patch) > 1): return None
    
    stain_patch = pull_stain_channel(image_patch, stain_vectors, channel=stain_channel)
        
    sitk_image  = sitk.GetImageFromArray(stain_patch.astype(np.uint8))
    sitk_mask   = sitk.GetImageFromArray(mask_patch.astype(np.uint8))

    try:
        bbox, _ = radiomics.imageoperations.checkMask(sitk_image, sitk_mask)
    except Exception as exc:
        logger.warning (f"Skipping this patch, mask pair due to '{exc}'")
    else:
        cimg, cmas = radiomics.imageoperations.cropToTumorMask(sitk_image, sitk_mask, bbox)

        fts = extractor.execute(sitk_image, sitk_mask, voxelBased=True)

        cluster_tend_patch   = sitk.GetArrayFromImage(fts[f'original_glcm_{glcm_feature}']).astype(np.float).flatten()
        cluster_tend_nonzero = cluster_tend_patch[cluster_tend_patch != 0]
        cluster_tend_valid   = cluster_tend_nonzero[~np.isnan(cluster_tend_nonzero)]

        return cluster_tend_valid
            
