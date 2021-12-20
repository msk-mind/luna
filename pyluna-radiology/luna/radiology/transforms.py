import os
import logging 

import numpy as np
import pandas as pd

from pathlib import Path
from dirhash import dirhash

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
from pydicom import dcmread
from medpy.io import load, save
import itk

from luna.radiology.mirp.importSettings        import Settings
from luna.radiology.mirp.imageReaders          import read_itk_image, read_itk_segmentation
from luna.radiology.mirp.imageProcess          import interpolate_image, interpolate_roi, crop_image
from luna.radiology.mirp.imagePerturbations    import randomise_roi_contours
from luna.radiology.mirp.imageProcess          import combine_all_rois, combine_pertubation_rois

class MatchRadiologyLabelFile:
    logger = logging.getLogger(__qualname__)
    def __init__(self):
        pass
    def __call__(self, tree_folder, input_label_data):
        dicom_folders = set()
        for dicom in Path(tree_folder).rglob("*.dcm"):
            dicom_folders.add(dicom.parent)
        
        label, label_header = load(input_label_data)
        found_dicom_path = None
        for dicom_folder in dicom_folders:
            n_slices_dcm = len(list((dicom_folder).glob("*.dcm")))
            if label.shape[2] == n_slices_dcm: found_dicom_path = dicom_folder

        if found_dicom_path: 
            path = next(found_dicom_path.glob("*.dcm"))
            ds = dcmread(path)
            print ("Matched: z=", label.shape[2], found_dicom_path, ds.PatientName, ds.AccessionNumber, ds.StudyDescription, ds.SeriesDescription, ds.SliceThickness)
        return str(found_dicom_path)


class DicomToITK:
    logger = logging.getLogger(__qualname__)
    def __init__(self, itk_image_type, ctype='signed short'):
        self.itk_image_type = itk_image_type
        self.ctype = ctype
    def __call__(self, input_path, output_dir):
        """
        Generate an ITK compatible image from a dicom series

        :param dicom_path: filepath to folder of dicom images
        :param output_dir: destination directory
        :param params {
            file_ext str: file extention for scan generation
        }

        :return: property dict, None if function fails
        """
        os.makedirs(output_dir, exist_ok=True)

        PixelType = itk.ctype('signed short')
        ImageType = itk.Image[PixelType, 3]

        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.AddSeriesRestriction("0008|0021")
        namesGenerator.SetGlobalWarningDisplay(False)
        namesGenerator.SetDirectory(input_path)

        seriesUIDs = namesGenerator.GetSeriesUIDs()
        num_dicoms = len(seriesUIDs)

        if num_dicoms < 1:
            self.logger.warning('No DICOMs in: ' + input_path)
            return None

        self.logger.info('The directory {} contains {} DICOM Series'.format(input_path, str(num_dicoms)))

        n_slices = 0

        for uid in seriesUIDs:
            self.logger.info('Reading: ' + uid)
            fileNames = namesGenerator.GetFileNames(uid)
            if len(fileNames) < 1: continue

            n_slices = len(fileNames)

            reader = itk.ImageSeriesReader[ImageType].New()
            dicomIO = itk.GDCMImageIO.New()
            reader.SetImageIO(dicomIO)
            reader.SetFileNames(fileNames)
            reader.ForceOrthogonalDirectionOff()

            writer = itk.ImageFileWriter[ImageType].New()

            outFileName = os.path.join(output_dir, uid + '_volumetric_image.' + self.itk_image_type)
            writer.SetFileName(outFileName)
            writer.UseCompressionOn()
            writer.SetInput(reader.GetOutput())
            self.logger.info('Writing: ' + outFileName)
            writer.Update()

        # Prepare metadata and commit
        properties = {
            'data' : outFileName,
            'zdim' : n_slices,
            'hash':  dirhash(output_dir, "sha256") 
        }

        return properties

class LesionRadiomicsExtractor:
    def __init__(self, lesion_indicies, enable_all_filters=False, check_geometry_strict=True, **kwargs):
        self.lesion_indicies = lesion_indicies
        self.enable_all_filters = enable_all_filters
        self.check_geometry_strict = check_geometry_strict
        self.kwargs = kwargs

    def __call__(self, input_image_data, input_label_data, output_dir):
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


        os.makedirs(output_dir, exist_ok=True)
        
        image, image_header = load(input_image_data)

        if   Path(input_label_data).is_dir():  label_path_list = [str(path) for path in Path(input_label_data).glob("*")]
        elif Path(input_label_data).is_file(): label_path_list = [input_label_data]
        else: raise RuntimeError("Issue with detecting label format")

        available_labels = set()
        for label_path in label_path_list:
            label, label_header = load(label_path)

            available_labels.update

            available_labels.update(np.unique(label))

            self.logger.info(f"Checking {label_path}")

            if self.check_geometry_strict and not image_header.get_voxel_spacing() == label_header.get_voxel_spacing(): 
                raise RuntimeError(f"Voxel spacing mismatch, image.spacing={image_header.get_voxel_spacing()}, label.spacing={label_header.get_voxel_spacing()}" )
            
            if not image.shape == label.shape:
                raise RuntimeError(f"Shape mismatch: image.shape={image.shape}, label.shape={label.shape}")

        df_result = pd.DataFrame()

        for lesion_index in available_labels.intersection(self.lesion_indicies):

            extractor = featureextractor.RadiomicsFeatureExtractor(label=lesion_index, **self.kwargs)

            for label_path in label_path_list:

                result = extractor.execute(input_image_data, label_path)

                result['lesion_index'] = lesion_index 
                result['diagnostics_image_data'] = input_image_data 
                result['diagnostics_label_data'] = label_path 

                df_result = pd.concat([df_result, pd.Series(result).to_frame()], axis=1)

        output_filename = os.path.join(output_dir, "radiomics.csv")

        df_result.T.to_csv(output_filename, index=False)

        properties = {
            'data' : output_filename,
            'lesion_indicies': self.lesion_indicies,
        }

        return properties
        
class WindowVolume:
    def __init__(self, low_level, high_level):
        self.low_level = low_level
        self.high_level = high_level

    def __call__(self, input_data, output_dir):

        os.makedirs(output_dir, exist_ok=True)
        
        file_stem = Path(input_data).stem
        file_ext  = Path(input_data).suffix

        outFileName = os.path.join(output_dir, file_stem + '.windowed' + file_ext)

        self.logger.info ("Applying window [%s,%s]", self.low_level, self.high_level)

        image, header = load(input_data)
        image = np.clip(image, self.low_level, self.high_level )
        save(image, outFileName, header)
        # Prepare metadata and commit
        properties = {
            'data' : outFileName,
        }

        return properties

class MirpProcessor:
    def __init__(self, resample_pixel_spacing, resample_beta):
        self.resample_pixel_spacing = resample_pixel_spacing
        self.resample_beta = resample_beta
    
    def __call__(self, input_image_data, input_label_data, output_dir):
        """
        Randomize contours given and image, label to and output_dir using MIRP processing library

        :param image_path: filepath to image
        :param label_path: filepath to 3d segmentation
        :param output_dir: destination directory
        :param params {

        }

        :return: property dict, None if function fails
        """
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info("Hello, processing %s, %s", input_image_data, input_label_data)
        settings = Settings()

        print (settings)

        settings.img_interpolate.new_spacing = self.resample_pixel_spacing
        settings.roi_interpolate.new_spacing = self.resample_pixel_spacing
        settings.img_interpolate.smoothing_beta = self.resample_beta

        # Read
        image_class_object      = read_itk_image(input_image_data, "CT")
        roi_class_object_list   = read_itk_segmentation(input_label_data)

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
    