import os, logging
from dirhash import dirhash

import numpy as np
import pandas as pd

from PIL import Image
import cv2
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
from pydicom import dcmread
from medpy.io import load, save
from skimage.transform import resize
import itk
from pathlib import Path

logger = logging.getLogger(__name__)

### Fix for annoying glcm message:
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

## Fix for ITK snap

def find_centroid(image, width, height):
    """
    Find the centroid of the 2d segmentation.

    :param image: segmentation slice as PIL image
    :param width: width of the image
    :param height: height of the image
    :return: (x, y) center point
    """
    seg = np.array(image)

    xcenter = np.argmax(np.mean(seg[:,:,0], axis=0))
    ycenter = np.argmax(np.mean(seg[:,:,0], axis=1))

    return (int(xcenter), int(ycenter))


def crop_images(xcenter, ycenter, dicom, overlay, crop_w, crop_h, image_w, image_h):
    """
    Crop PNG images around the centroid (xcenter, ycenter).

    :param xcenter: x center point to crop around. result of find_centroid()
    :param ycenter: y center point to crop around. result of find_centroid()
    :param dicom: dicom PIL image
    :param overlay: overlay PIL image
    :param crop_w: desired width of cropped image
    :param crop_h: desired height of the cropped image
    :param image_w: width of the original image
    :param image_h: height of the original image
    :return: binary tuple (dicom, overlay)
    """
    crop_w, crop_h = int(crop_w), int(crop_h)
    image_w, image_h = int(image_w), int(image_h)
    # Find xmin, ymin, xmax, ymax based on CROP_SIZE
    width_rad = crop_w // 2
    height_rad = crop_h // 2

    xmin, ymin, xmax, ymax = (xcenter - width_rad), (ycenter - height_rad), (xcenter + width_rad), (ycenter + height_rad)

    if xmin < 0:
        xmin = 0
        xmax = crop_w

    if xmax > image_w:
        xmin = image_w - crop_w
        xmax = image_w

    if ymin < 0:
        ymin = 0
        ymax = crop_h

    if ymax > image_h:
        ymin = image_h - crop_h
        ymax = image_h

    # Crop overlay, dicom pngs.
    dicom_feature = dicom.crop((xmin, ymin, xmax, ymax))
    overlay_feature = overlay.crop((xmin, ymin, xmax, ymax))

    # normalize after crop
    dicom_feature = normalize(np.array(dicom_feature)).tobytes()
    overlay_feature = normalize(np.array(overlay_feature)).tobytes()

    return (dicom_feature, overlay_feature)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize scan image intensity. Sets minimum value to zero, rescales by
    alpha factor and casts to uint8 w/ saturation.
    :param np.ndarray image: a single slice of an mr scan
    :return np.ndarray normalized_image: normalized mr slice
    """

    image = image - np.min(image)

    alpha_norm = 255.0 / min(np.max(image) - np.min(image), 10000)

    normalized_image = cv2.convertScaleAbs(image, alpha=alpha_norm)

    return normalized_image

def slice_to_image(image_slice, width, height, normalize=False):
    """
    Normalize and create an image binary from the given 2D array.

    :param image_slice: image slice
    :param width: width of the image
    :param height: height of the image
    :param normalize: if True normalize the image.
    :return: PIL image
    """
    from preprocess import normalize

    # Convert 2d image to float to avoid overflow or underflow losses.
    # Transpose to get the preserve x, y coordinates.
    image_2d = image_slice.astype(float).T

    # Rescaling grey scale between 0-255
    if normalize:
        image_2d = normalize(image_2d)

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d)

    im = Image.fromarray(image_2d_scaled)
    # resize pngs to user provided width/height
    im = im.resize( (width, height) )

    return im


def subset_bound_seg(src_path, output_path, start_slice, end_slice):
    """
    Pull out desired range of slices from segmentations created from
    a bound scan (where multiple scans are bound in one series)

    :param src_path: path to a segmentation file
    :param output_path: path to new segmentation file
    :param start_slice: starting slice
    :param end_slice:  ending slice
    :return: new segmentation file path
    """
    start_slice = int(start_slice)
    end_slice = int(end_slice)
    try:
        file_path = src_path.split(':')[-1]
        data, header = load(file_path)
        subset = data[:,:,start_slice:end_slice]
        save(subset, output_path, hdr=header)
    except Exception as err:
        print(err)
        return None
    return output_path


def create_images(scan_path, seg_path, subset_scan_path, subset_seg_path,
                      width, height, crop_width, crop_height, n_slices=None):
    """
    Create images from 3d segmentations.

    :param scan_path: filepath to 3d series
    :param seg_path: filepath to 3d segmentation
    :param subset_scan_path: filepath to 3d segmentation
    :param subset_seg_path: filepath to 3d series
    :param width: width of the image
    :param height: height of the image
    :param crop_width: optional width of the image
    :param crop_height: optional height of the image
    :param n_slices: optionally provide n_slices to return.
    The subset will be taken from the middle
    :return: an array of (n_tumor_slices, dicom_binary, overlay_binary) tuples
    """
    from preprocess import normalize

    if subset_seg_path:
        file_path = subset_seg_path.split(':')[-1]
    else:
        file_path = seg_path.split(':')[-1]
    print("Processing ", file_path)
    data, header = load(file_path)
    num_images = data.shape[2]

    # Find the annotated slices with 3d segmentation.
    # Some reverse engineering.. save the instance numbers
    # from the series to identify the dicom slices that were annotated.
    slices = []
    for i in range(num_images):
        image_slice = data[:,:,i]

        if np.any(image_slice):
            im = slice_to_image(image_slice, width, height)

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)

            slices.append( (i, rgb) )

    if len(slices) == 0:
        print("No annotation found ", file_path)
        return None

    slices_len = len(slices)
    mid_idx = slices_len//2
    # find centroid using the mid segmentation and return x,y
    centroid = find_centroid(slices[mid_idx][1], width, height)
    
    res = [slice + (slices_len, centroid[0], centroid[1]) for slice in slices]

    # if the user specified n_slices to select, then select the n_slices from the middle.
    if n_slices and n_slices < slices_len:
        before = n_slices//2
        after = n_slices - before
        res = res[mid_idx - before: mid_idx + after]

    ## populate SCAN images for indices identified from SEG processing.
    if subset_scan_path:
        file_scan_path = subset_scan_path.split(':')[-1]
    else:
        file_scan_path = scan_path.split(':')[-1]
    print("Processing ", file_scan_path)
    data, header = load(file_scan_path)

    scans = []
    for res_slice in res:
        image_slice = data[:,:,res_slice[0]]
        #image_slice = np.flipud(data[:,:,res_slice[0]])
        if crop_width and crop_height:
            im = slice_to_image(image_slice, width, height)
        else:
            im = slice_to_image(image_slice, width, height, normalize=True)
        scans.append(im)

    images = []
    for idx in range(len(res)):
        # load dicom and seg images from bytes
        dicom_img = scans[idx]
        dicom_binary = dicom_img.tobytes()
        seg_img = res[idx][1]

        overlay = Image.blend(dicom_img.convert("RGB"), seg_img, 0.3)

        if crop_width and crop_height:
            dicom_binary, overlay = crop_images(res[idx][3], res[idx][4], dicom_img, overlay, crop_width, crop_height, width, height)
        else:
            overlay = overlay.tobytes()
        images.append((res[idx][2], dicom_binary, overlay))

    return images


def calculate_target_shape(volume, header, target_spacing):
    """
    Calculates a new number of pixels along a dimension determined by multiplying the 
    current dimension by a scale factor of (source spacing / target spacing)

    The dimension of the volume, header spacing, and target spacing must all match, but the common usecase is 3D

    :param volume: as numpy.ndarray
    :param header: ITK-SNAP header
    :param target_spacing: as tuple or list
    :return: target_shape as list
    """
    src_spacing = header.get_voxel_spacing()
    target_shape = [int(src_d * src_sp / tar_sp) for src_d, src_sp, tar_sp in
                    zip(volume.shape, src_spacing, target_spacing)]
    return target_shape

def resample_volume(volume, order, target_shape):
    """
    Resamples volume using order specified, returning a new volume with resized dimensions = target_shape

    :param volume: as numpy.ndarray
    :param order: 0 for NN (for segmentation), 3 for cubic (recommended for acquisition)
    :param target_shape: as tuple or list
    :return: Resampled volume as numpy.ndarray
    """
    # Only anti_alias if order =/= 0
    anti_alias = False if order == 0 else True

    volume = resize(volume, target_shape,
                    order=order, clip=True, mode='edge',
                    preserve_range=True, anti_aliasing=anti_alias)
    return volume

def interpolate_segmentation_masks(seg, target_shape):
    """
    Use NN interpolation for segmentation masks by resampling boolean masks for each value present.
    :param seg: as numpy.ndarray
    :param target_shape: as tuple or list
    :return: new segmentation as numpy.ndarray
    """
    new_seg = np.zeros(target_shape).astype(int)
    for roi in np.unique(seg):
        if roi == 0:
            continue
        mask = resample_volume(seg == roi, 0, target_shape).astype(bool)
        new_seg[mask] = int(roi)
    return new_seg


