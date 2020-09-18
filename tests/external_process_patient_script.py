import os, sys, subprocess, time

# Test Imports:

'''
This was used to test whether c++ libraries would run without pointer/pickling errors when called through a pandasUDF. 
This was tested by importing the openslide library and was successful. No need to import these libraries.
'''
# import openslide

'''
This was to show that libraries not imported in the original preprocess_feature script would still be able to be imported 
successfully when called through a pandas UDF
'''
# import databricks.koalas as ks



import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from skimage.transform import resize

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, expr
from pyspark.sql.types import StringType




GPFS_MOUNT_DIR = "/gpfs/mskmindhdp_emc"


def generate_absolute_path_from_hdfs(absolute_hdfs_path_col, filename_col):
    # do we need this if?
    GPFS_MOUNT_DIR = "/gpfs/mskmindhdp_emc"
    if absolute_hdfs_path_col[0] == '/':
        absolute_hdfs_path_col = absolute_hdfs_path_col[1:]
    return os.path.join(GPFS_MOUNT_DIR, absolute_hdfs_path_col, filename_col)


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


def generate_preprocessed_filename(id, suffix, processed_dir, target_spacing_x, target_spacing_y, target_spacing_z):
    """
    Generates target NumPy file path for preprocessed segmentation or acquisition.
    :param idx: case ID
    :param suffix: _seg or _img, depending on which Series is being populated.
    :param processed_dir: path to save .npy files.
    :param target_spacing_x  target x-dimension spacing
    :param target_spacing_y target y-dimension spacing
    :param target_spacing_z target z-dimension spacing
    :return: target file path
    """
    file_name = "".join((processed_dir, str(id), suffix, "_", str(target_spacing_x), "_", str(target_spacing_y), "_", str(target_spacing_z), ".npy"))
    return file_name


def calculate_target_shape(volume, header, target_spacing):
    """
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
    Resamples volume using order specified.
    :param volume: as numpy.ndarray
    :param order: 0 for NN (for segmentation), 3 for cubic (recommended for acquisition)
    :param target_shape: as tuple or list
    :return: Resampled volume as numpy.ndarray
    """
    if order == 0:
        anti_alias = False
    else:
        anti_alias = True

    volume = resize(volume, target_shape,
                    order=order, clip=True, mode='edge',
                    preserve_range=True, anti_aliasing=anti_alias)
    return volume

def generate_mhd_test():
    #!/usr/bin/env python
    import os
    import sys
    import itk


    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] +
            " [DicomDirectory [outputFileName [seriesName]]]")
        print("If DicomDirectory is not specified, current directory is used\n")

    # current directory by default
    dirName = '.'
    if len(sys.argv) > 1:
        dirName = sys.argv[1]

    PixelType = itk.ctype('signed short')
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dirName)

    seriesUID = namesGenerator.GetSeriesUIDs()

    if len(seriesUID) < 1:
        print('No DICOMs in: ' + dirName)
        sys.exit(1)

    print('The directory: ' + dirName)
    print('Contains the following DICOM Series: ')
    for uid in seriesUID:
        print(uid)

    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid
        if len(sys.argv) > 3:
            seriesIdentifier = sys.argv[3]
            seriesFound = True
        print('Reading: ' + seriesIdentifier)
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)

        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
        reader.ForceOrthogonalDirectionOff()

        writer = itk.ImageFileWriter[ImageType].New()
        outFileExt = 'mhd'
        if len(sys.argv) > 2:
            outFileExt = sys.argv[2]
        outFileName = os.path.join(dirName, "outputs", seriesIdentifier + '.' + outFileExt)
        writer.SetFileName(outFileName)
        writer.UseCompressionOn()
        writer.SetInput(reader.GetOutput())
        print('Writing: ' + outFileName)
        writer.Update()

        if seriesFound:
            break


def process_patient(patient: pd.DataFrame) -> pd.DataFrame:
    """
        Given a row with source and destination file paths for a single case, resamples segmentation
        and acquisition. Also, clips acquisition range to abdominal window.
        :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
        :return: None
    """
    scan_absolute_hdfs_path = generate_absolute_path_from_hdfs(patient.scan_absolute_hdfs_path.item(), patient.scan_filename.item())
    print("scan path", scan_absolute_hdfs_path)
    annotation_absolute_hdfs_path = generate_absolute_path_from_hdfs(patient.annotation_absolute_hdfs_path.item(), patient.annotation_filename.item())
    print("annot path", annotation_absolute_hdfs_path)
    preprocessed_scan_path = patient.preprocessed_scan_path.item()
    preprocessed_annotation_path = patient.preprocessed_annotation_path.item()   
    target_spacing = (patient.preprocessed_target_spacing_x, patient.preprocessed_target_spacing_y, patient.preprocessed_target_spacing_z)

    if os.path.exists(preprocessed_scan_path) and os.path.exists(preprocessed_annotation_path):
        print(preprocessed_scan_path + " and " + preprocessed_annotation_path + " already exists.")
        print(preprocessed_scan_path + " and " + preprocessed_annotation_path + " already exists.")
        return patient

    if not os.path.exists(scan_absolute_hdfs_path):
        print(scan_absolute_hdfs_path + " does not exist.")
        print(scan_absolute_hdfs_path + " does not exist.")
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""
        return patient

    if not os.path.exists(annotation_absolute_hdfs_path):
        print(annotation_absolute_hdfs_path + " does not exist.")
        print(annotation_absolute_hdfs_path + " does not exist.")
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""
        return patient

    try: 
        img, img_header = load(scan_absolute_hdfs_path)
        target_shape = calculate_target_shape(img, img_header, target_spacing)

        img = resample_volume(img, 3, target_shape)
        np.save(preprocessed_scan_path, img)
        print("saved img at " + preprocessed_scan_path)
        print("saved img at " + preprocessed_scan_path)

        seg, _ = load(annotation_absolute_hdfs_path)
        seg = interpolate_segmentation_masks(seg, target_shape)
        np.save(preprocessed_annotation_path, seg)
        print("saved seg at " + preprocessed_annotation_path)
        print("saved seg at " + preprocessed_annotation_path)
    except Exception as err:
        print("failed to generate resampled volume.", err)
        print("failed to generate resampled volume.", err)
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""

    return patient