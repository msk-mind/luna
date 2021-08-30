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


def subset_bound_dicom(src_path, output_path, index):
    """
    Pull out a from bound series, where the image array has dimensions like
    (x, y, n_bound_series, z)

    :param src_path: path to nifti file
    :param output_path: path to new dicom series file
    :param index: index to subset. should be less than n_bound_series
    :return: path to new dicom series file
    """
    index = int(index)
    try:
        file_path = src_path.split(':')[-1]
        data, header = load(file_path)
        subset = data[:,:,index,:]
        # re-arrange the array
        subset = np.swapaxes(np.swapaxes(subset, 1,2), 0,1)
        save(np.fliplr(subset), output_path)
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


def generate_scan(dicom_path: str, output_dir: str, params: dict, tag=None) -> dict:
    """
    Generate an ITK compatible image from a dicom series

    :param dicom_path: filepath to folder of dicom images
    :param output_dir: destination directory
    :param params {
        file_ext str: file extention for scan generation
    }

    :return: property dict, None if function fails
    """

    if tag is not None:
        output_dir =  os.path.join(output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)

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
        'data' : output_dir + "/" + uid + '_volumetric_image.' + params['itkImageType'],
        'zdim' : n_slices,
        'hash':  dirhash(output_dir, "sha256") 
    }

    if params['itkImageType']=='mhd':
        properties['aux'] =  output_dir + "/" + uid + '_volumetric_image.zraw'
      

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
        "data":output_dir, 
        "hash":dirhash(output_dir, "sha256")
    }

    return properties

def extract_radiomics(image_path: str, label_path: str, output_dir: str, params: dict, tag=None) -> dict:
    """
    Extract radiomics given and image, label to and output_dir, parameterized by params

    :param image_path: filepath to image
    :param label_path: filepath to 3d segmentation(s) as single path or list
    :param output_dir: destination directory
      :param params {
        radiomicsFeatureExtractor dict: configuration for the RadiomicsFeatureExtractor
        enableAllImageTypes bool: flag to enable all image types
    }

    :return: property dict, None if function fails
    """
    lesion_index = params['radiomicsFeatureExtractor']['label']

    if tag is not None:
        output_dir =  os.path.join(output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Image data: %s", image_path)
    logger.info("Label data: %s", label_path)

    if   Path(label_path).is_dir():  label_path_list = [str(path) for path in Path(label_path).glob("*")]
    elif Path(label_path).is_file(): label_path_list = [label_path]

    result_list = []
    for label_path in label_path_list:
        image, image_header = load(image_path)
        label, label_header = load(label_path)

        if params['radiomicsFeatureExtractor']['label'] not in label: 
            logger.warning(f"No mask pixels labeled [{params['radiomicsFeatureExtractor']['label']}] found, returning None")
            return None 

        extractor = featureextractor.RadiomicsFeatureExtractor(**params.get('radiomicsFeatureExtractor', {}))

        if params.get("strictGeometry", False): 
            if not image_header.get_voxel_spacing() == label_header.get_voxel_spacing():
                raise RuntimeError(f"Voxel spacing mismatch, image.spacing={image_header.get_voxel_spacing()}, label.spacing={label_header.get_voxel_spacing()}" )
        
        if not image.shape == label.shape:
            raise RuntimeError(f"Shape mismatch: image.shape={image.shape}, label.shape={label.shape}")

        if params.get("enableAllImageTypes", False): extractor.enableAllImageTypes()

        result = extractor.execute(image_path, label_path)
        result['lesion_index'] = lesion_index 
        result['job_tag'] = tag 

        result_list.append( pd.Series(result).to_frame().transpose() )

    output_filename = os.path.join(output_dir, f"tag{tag}-lesion_index{lesion_index}.csv")
    logger.info("Saving to " + output_filename)
    pd.concat(result_list).to_csv(output_filename)

    logger.info(pd.concat(result_list))

    return pd.concat(result_list).set_index(['job_tag', 'lesion_index'])

    # return properties


def window_dicoms(dicom_path: str, output_dir: str, params: dict, tag=None) -> dict:
    """
    Extract radiomics given and image, label to and output_dir, parameterized by params

    :param dicom_paths: path to folder of dicoms to process
    :param output_dir: destination directory
    :param params {
        window bool: whether to apply windowing
        windowLowLevel int, float : lower level to clip
        windowHighLevel int, float: higher level to clip
    }

    :return: property dict, None if function fails
    """ 

    if tag is not None:
        output_dir =  os.path.join(output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)

    if params.get('window', False):
        logger.info ("Applying window [%s,%s]", params['windowLowLevel'], params['windowHighLevel'])

    for dcm in Path(dicom_path).glob("*.dcm"):
        ds = dcmread(dcm)
        hu = ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept
        ds.RescaleSlope = 0.0
        ds.RescaleIntercept = 0.0
        if params['window']:
            hu = np.clip( hu, params['windowLowLevel'], params['windowHighLevel']   )
        ds.PixelData = hu.astype(ds.pixel_array.dtype).tobytes()
        ds.save_as (os.path.join( output_dir, dcm.stem + ".cthu.dcm"  ))

    # Prepare metadata and commit
    properties = {
        "RescaleSlope": float(ds.RescaleSlope), 
        "RescaleIntercept": float(ds.RescaleIntercept), 
        "units": "HU", 
        "data": output_dir, 
        "hash": dirhash(output_dir, "sha256")
    }

    return properties


from luna.radiology.mirp.importSettings        import Settings
from luna.radiology.mirp.imageReaders          import read_itk_image, read_itk_segmentation
from luna.radiology.mirp.imageProcess          import interpolate_image, interpolate_roi, crop_image
from luna.radiology.mirp.imagePerturbations    import randomise_roi_contours
from luna.radiology.mirp.imageProcess          import combine_all_rois, combine_pertubation_rois

def randomize_contours(image_path: str, label_path: str, output_dir: str, params: dict, tag=None) -> dict:
    """
    Randomize contours given and image, label to and output_dir using MIRP processing library

    :param image_path: filepath to image
    :param label_path: filepath to 3d segmentation
    :param output_dir: destination directory
      :param params {

    }

    :return: property dict, None if function fails
    """
    if tag is not None:
        output_dir =  os.path.join(output_dir, tag)
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Hello, processing %s, %s", image_path, label_path)
    settings = Settings()

    settings.img_interpolate.new_spacing = params['mirpResampleSpacing']
    settings.roi_interpolate.new_spacing = params['mirpResampleSpacing']
    settings.img_interpolate.smoothing_beta = params['mirpResampleBeta']

    # Read
    image_class_object      = read_itk_image(image_path, "CT")
    roi_class_object_list   = read_itk_segmentation(label_path)

    # Crop for faster interpolation
    image_class_object, roi_class_object_list = crop_image(img_obj=image_class_object, roi_list=roi_class_object_list, boundary=50.0, z_only=True)

    # Interpolation
    image_class_object    = interpolate_image (img_obj=image_class_object, settings=settings)
    roi_class_object_list = interpolate_roi   (img_obj=image_class_object, roi_list=roi_class_object_list, settings=settings)

    # Export
    image_file = image_class_object.export(file_path=f"{output_dir}/main_image")

    # ROI processing
    roi_class_object = combine_all_rois (roi_list=roi_class_object_list, settings=settings)
    label_file = roi_class_object.export(img_obj=image_class_object, file_path=f"{output_dir}/main_label")

    roi_class_object_list, svx_class_object_list = randomise_roi_contours (img_obj=image_class_object, roi_list=roi_class_object_list, settings=settings)

    roi_supervoxels = combine_all_rois (roi_list=svx_class_object_list, settings=settings)
    voxels_file = roi_supervoxels.export(img_obj=image_class_object, file_path=f"{output_dir}/supervoxels")

    for roi in combine_pertubation_rois (roi_list=roi_class_object_list, settings=settings): 
        if "COMBINED" in roi.name: roi.export(img_obj=image_class_object, file_path=f"{output_dir}/pertubations")

    print (image_file, label_file)

    # Construct return dicts
    main_image_properties       = {"data": image_file}
    main_label_properties       = {"data": label_file}
    pertubation_set_properties  = {"data": f"{output_dir}/pertubations"}
    supervoxel_properties       = {"data": voxels_file}
    return main_image_properties, main_label_properties, pertubation_set_properties, supervoxel_properties
