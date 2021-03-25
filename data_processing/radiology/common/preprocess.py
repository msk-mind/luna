import os, logging
from dirhash import dirhash

import numpy as np
import pandas as pd

from PIL import Image
from medpy.io import load
import cv2
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
from pydicom import dcmread
from medpy.io import load
from skimage.transform import resize
import itk

def find_centroid(path, image_w, image_h):
    """
    Find the centroid of the 2d segmentation.

    :param path: filepath to 2d segmentation file
    :param image_w: width of the image
    :param image_h: height of the image
    :return: (x, y) center point
    """

    # 2d segmentation file path
    file_path = path.split(':')[-1]
    data, header = load(file_path)

    h, w, num_images = data.shape

    # Find the annotated slice
    xcenter, ycenter = 0, 0
    for i in range(num_images):
        seg = data[:,:,i]
        if np.any(seg):
            print(i)
            seg = seg.astype(float)

            # find centroid using mean
            xcenter = np.argmax(np.mean(seg, axis=1))
            ycenter = np.argmax(np.mean(seg, axis=0))
            break

    # Check if h,w matches IMAGE_WIDTH, IMAGE_HEIGHT. If not, this is due to png being rescaled. So scale centers.
    image_w, image_h = int(image_w), int(image_h)
    if not h == image_h:
        xcenter = int(xcenter * image_w // w)
    if not w == image_w:
        ycenter = int(ycenter * image_h // h)

    return (int(xcenter), int(ycenter))


def crop_images(xcenter, ycenter, dicom, overlay, crop_w, crop_h, image_w, image_h):
    """
    Crop PNG images around the centroid (xcenter, ycenter).

    :param xcenter: x center point to crop around. result of find_centroid()
    :param ycenter: y center point to crop around. result of find_centroid()
    :param dicom: dicom binary data
    :param overlay: overlay binary data
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
    dicom_img = Image.frombytes("L", (image_w, image_h), bytes(dicom))
    dicom_feature = dicom_img.crop((xmin, ymin, xmax, ymax)).tobytes()

    overlay_img = Image.frombytes("RGB", (image_w, image_h), bytes(overlay))
    overlay_feature = overlay_img.crop((xmin, ymin, xmax, ymax)).tobytes()

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

def dicom_to_bytes(dicom_path, width, height):
    """
    Create an image binary from dicom image.

    :param dicom_path: filepath to dicom
    :param width: width of the image
    :param height: height of the image
    :return: image in bytes
    """
    from preprocess import normalize

    file_path = dicom_path.split(':')[-1]

    data, header = load(file_path)

    # Convert 2d image to float to avoid overflow or underflow losses.
    # Transpose to get the preserve x, y coordinates.
    image_2d = data[:,:,0].astype(float).T

    # Rescaling grey scale between 0-255
    image_2d_scaled = normalize(image_2d)

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    im = Image.fromarray(image_2d_scaled)
    # resize pngs to user provided width/height
    im = im.resize( (width, height) )

    return im.tobytes()


def create_seg_images(src_path, uuid, width, height):
    """
    Create images from 3d segmentations.

    :param src_path: filepath to 3d segmentation
    :param uuid: scan uuid
    :param width: width of the image
    :param height: height of the image
    :return: an array of (instance_number, uuid, png binary) tuples
    """
    from preprocess import normalize

    file_path = src_path.split(':')[-1]
    data, header = load(file_path)

    num_images = data.shape[2]

    # Find the annotated slices with 3d segmentation.
    # Some reverse engineering.. save the instance numbers
    # from the series to identify the dicom slices that were annotated.
    slices = []
    for i in range(num_images):
        image_slice = data[:,:,i]
        if np.any(image_slice):
            image_2d = image_slice.astype(float).T
            # double check that subtracting is needed for all.
            slice_num = num_images - (i+1)

            image_2d_scaled = normalize(image_2d)
            image_2d_scaled = np.uint8(image_2d_scaled)

            im = Image.fromarray(image_2d_scaled)
            # resize pngs to user provided width/height
            im = im.resize( (int(width), int(height)) )

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)
            png_binary = rgb.tobytes()

            slices.append( (slice_num, uuid, png_binary) )

    return slices


def overlay_images(dicom_path, seg, width, height):
    """
    Create dicom images.
    Create overlay images by blending dicom and segmentation images with 7:3 ratio.

    :param dicom_path: filepath to the dicom file
    :param seg: segmentation image in bytes
    :param width: width of the image
    :param height: height of the image
    :return: (dicom, overlay) tuple of binaries
    """
    width, height = int(width), int(height)
    dicom_binary = dicom_to_bytes(dicom_path, width, height)

    # load dicom and seg images from bytes
    dcm_img = Image.frombytes("L", (width, height), bytes(dicom_binary))
    dcm_img = dcm_img.convert("RGB")
    seg_img = Image.frombytes("RGB", (width, height), bytes(seg))

    res = Image.blend(dcm_img, seg_img, 0.3)
    overlay = res.tobytes()

    return (dicom_binary, overlay)

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


def generate_scan(dicom_path: str, output_dir: str, params: dict) -> dict:
    """
    Generate an ITK compatible image from a dicom series

    :param dicom_path: filepath to folder of dicom images
    :param output_dir: destination directory
    :param params {
        file_ext str: file extention for scan generation
    }

    :return: property dict, None if function fails
    """
    logger = logging.getLogger(__name__)

    PixelType = itk.ctype('signed short')
    ImageType = itk.Image[PixelType, 3]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dicom_path)

    seriesUIDs = namesGenerator.GetSeriesUIDs()
    num_dicoms = len(seriesUIDs)

    if num_dicoms < 1:
        logger.warning('No DICOMs in: ' + dicom_path)
        return None

    logger.info('The directory {} contains {} DICOM Series'.format(dicom_path, str(num_dicoms)))

    n_slices = 0

    for uid in seriesUIDs:
        logger.info('Reading: ' + uid)
        fileNames = namesGenerator.GetFileNames(uid)
        if len(fileNames) < 1: continue

        n_slices = len(fileNames)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()

        outFileName = os.path.join(output_dir, uid + '_volumetric_image.' + params['itkImageType'])
        writer.SetFileName(outFileName)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())
        logger.info('Writing: ' + outFileName)
        writer.Update()

    # Prepare metadata and commit
    properties = {
        'path' : output_dir,
        'zdim' : n_slices,
        'hash':  dirhash(output_dir, "sha256") 
    }

    return properties

def extract_voxels(image_path: str, label_path: str, output_dir: str, params: dict) -> dict:
    """
    Perform resampling on an input image and mask, and save as binary .npy files

    :param image_path: filepath to image
    :param label_path: filepath to 3d segmentation
    :param output_dir: destination directory
      :param params {
        resampledPixelSpacing dict: configuration for the RadiomicsFeatureExtractor
        enableAllImageTypes bool: flag to enable all image types
    }

    :return: property dict, None if function fails
    """
    logger = logging.getLogger(__name__)

    img, img_header = load(image_path)
    seg, seg_header = load(label_path)

    logger.info("Extracting voxels with resampledPixelSpacing = %s", params['resampledPixelSpacing'])

    target_shape = calculate_target_shape(img, img_header, params['resampledPixelSpacing'])

    logger.info("Target shape = %s", target_shape)

    img_resampled = resample_volume(img, 3, target_shape)
    logger.info("Resampled image with size %s", img_resampled.shape)
    img_output_filename = os.path.join(output_dir, "image_voxels.npy")
    np.save (img_output_filename, img_resampled)
    logger.info("Saved resampled image at %s", img_output_filename)

    seg_interpolated = interpolate_segmentation_masks(seg, target_shape)
    logger.info("Resampled segmentation with size %s", seg_interpolated.shape)
    seg_output_filename = os.path.join(output_dir, "label_voxels.npy")
    np.save(seg_output_filename, seg_interpolated)
    logger.info("Saved resampled mask at %s", seg_output_filename)

    # Prepare metadata and commit
    properties = {
        "resampledPixelSpacing":params['resampledPixelSpacing'], 
        "targetShape": target_shape,
        "path":output_dir, 
        "hash":dirhash(output_dir, "sha256")
    }

    return properties

def extract_radiomics(image_path: str, label_path: str, output_dir: str, params: dict) -> dict:
    """
    Extract radiomics given and image, label to and output_dir, parameterized by params

    :param image_path: filepath to image
    :param label_path: filepath to 3d segmentation
    :param output_dir: destination directory
      :param params {
        RadiomicsFeatureExtractor dict: configuration for the RadiomicsFeatureExtractor
        enableAllImageTypes bool: flag to enable all image types
    }

    :return: property dict, None if function fails
    """
    logger = logging.getLogger(__name__)

    extractor = featureextractor.RadiomicsFeatureExtractor(**params.get('RadiomicsFeatureExtractor', {}))

    if params.get("strictGeometry", False): 
        image, image_header = load(image_path)
        label, label_header = load(label_path)
        if not image_header.get_voxel_spacing() == label_header.get_voxel_spacing():
            raise RuntimeError(f"Voxel spacing mismatch, image.spacing={image_header.get_voxel_spacing()}, label.spacing={label_header.get_voxel_spacing()}" )
        if not image.shape == label.shape:
            raise RuntimeError(f"Shape mismatch: image.shape={image.shape}, label.shape={label.shape}")


    if params.get("enableAllImageTypes", False): extractor.enableAllImageTypes()

    result = extractor.execute(image_path, label_path)

    output_filename = os.path.join(output_dir, "radiomics-out.csv")

    logger.info("Saving to " + output_filename)
    sers = pd.Series(result)
    sers.to_frame().transpose().to_csv(output_filename)

    # Prepare metadata and commit
    properties = {
        "path":output_dir, 
        "hash":dirhash(output_dir, "sha256")
    }

    return properties


def window_dicoms(dicom_paths: list, output_dir: str, params: dict) -> dict:
    """
    Extract radiomics given and image, label to and output_dir, parameterized by params

    :param dicom_paths: list of filepaths to process
    :param output_dir: destination directory
    :param params {
        window bool: whether to apply windowing
        window_low_level  int, float : lower level to clip
        window_high_level int, float: higher level to clip
    }

    :return: property dict, None if function fails
    """ 
 
    logger = logging.getLogger(__name__)

    # Scale and clip each dicom, and save in new directory
    logger.info("Processing %s dicoms!", len(dicom_paths))
    if params.get('window', False):
        logger.info ("Applying window [%s,%s]", params['window_low_level'], params['window_high_level'])
    for dcm in dicom_paths:
        ds = dcmread(dcm)
        hu = ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept
        if params['window']:
            hu = np.clip( hu, params['window_low_level'], params['window_high_level']   )
        ds.PixelData = hu.astype(ds.pixel_array.dtype).tobytes()
        ds.save_as (os.path.join( output_dir, dcm.stem + ".cthu.dcm"  ))

    # Prepare metadata and commit
    properties = {
        "RescaleSlope": float(ds.RescaleSlope), 
        "RescaleIntercept": float(ds.RescaleIntercept), 
        "units": "HU", 
        "path": output_dir, 
        "hash": dirhash(output_dir, "sha256")
    }

    return properties


from data_processing.radiology.mirp.importSettings        import Settings
from data_processing.radiology.mirp.imageReaders          import read_itk_image, read_itk_segmentation
from data_processing.radiology.mirp.imageProcess          import interpolate_image, interpolate_roi, crop_image
from data_processing.radiology.mirp.imagePerturbations    import randomise_roi_contours
from data_processing.radiology.mirp.imageProcess          import combine_all_rois, combine_pertubation_rois

def randomize_contours(image_path: str, label_path: str, output_dir: str, params: dict) -> dict:
    """
    Randomize contours given and image, label to and output_dir using MIRP processing library

    :param image_path: filepath to image
    :param label_path: filepath to 3d segmentation
    :param output_dir: destination directory
      :param params {

    }

    :return: property dict, None if function fails
    """
    logger = logging.getLogger(__name__)
    logger.info("Hello, processing %s, %s", image_path, label_path)
    settings = Settings()

    # Read
    image_class_object      = read_itk_image(image_path, "CT")
    roi_class_object_list   = read_itk_segmentation(label_path)

    # Crop for faster interpolation
    image_class_object, roi_class_object_list = crop_image(img_obj=image_class_object, roi_list=roi_class_object_list, boundary=50.0, z_only=True)

    # Interpolation
    image_class_object    = interpolate_image (img_obj=image_class_object, settings=settings)
    roi_class_object_list = interpolate_roi   (img_obj=image_class_object, roi_list=roi_class_object_list, settings=settings)

    # Export
    image_class_object.export(file_path=f"{output_dir}/main_image")

    # ROI processing
    for roi in combine_all_rois (roi_list=roi_class_object_list, settings=settings): 
        if "COMBINED" in roi.name: roi.export(img_obj=image_class_object, file_path=f"{output_dir}/main_label")

    roi_class_object_list, svx_class_object_list = randomise_roi_contours (img_obj=image_class_object, roi_list=roi_class_object_list, settings=settings)
    for roi in combine_all_rois (roi_list=svx_class_object_list, settings=settings):
        if "SUPERVOXEL" in roi.name: roi.export(img_obj=image_class_object, file_path=f"{output_dir}/supervoxels")

    for roi in combine_pertubation_rois (roi_list=roi_class_object_list, settings=settings): 
        if "COMBINED" in roi.name: roi.export(img_obj=image_class_object, file_path=f"{output_dir}/pertubations")

    # Construct return dicts
    main_image_properties = {"path":f"{output_dir}/main_image"}
    main_label_properties = {"path":f"{output_dir}/main_label"}
    pertubation_set_properties = {"path":f"{output_dir}/pertubations"}
    supervoxel_properties = {"path":f"{output_dir}/supervoxels"}
    return main_image_properties, main_label_properties, pertubation_set_properties, supervoxel_properties
