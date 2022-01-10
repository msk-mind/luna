import os, logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('window_volume')

from luna.common.utils import cli_runner

_params_ = [('input_image_data', str), ('input_label_data', str), ('resample_pixel_spacing', float), ('resample_smoothing_beta', float), ('output_dir', str)]

@click.command()
@click.option('-ii', '--input_image_data', required=False,
              help='path to input image data')
@click.option('-il', '--input_label_data', required=False,
              help='path to input label data')
@click.option('-rps', '--resample_pixel_spacing', required=False,
              help='path to input label data')
@click.option('-rsb', '--resample_smoothing_beta', required=False,
              help='path to input label data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Randomize contours given and image, label to and output_dir using MIRP processing library

    \b
        coregister_volumes
            --input_image_data volume_ct.nii
            --input_label_data labels.nii
            -rps 1.5
            -rsb 0.9
            -o ./mirp_results/
    """
    cli_runner(cli_kwargs, _params_, randomize_contours )



from luna.radiology.mirp.importSettings        import Settings
from luna.radiology.mirp.imageReaders          import read_itk_image, read_itk_segmentation
from luna.radiology.mirp.imageProcess          import interpolate_image, interpolate_roi, crop_image
from luna.radiology.mirp.imagePerturbations    import randomise_roi_contours
from luna.radiology.mirp.imageProcess          import combine_all_rois, combine_pertubation_rois

import numpy as np

def randomize_contours(input_image_data, input_label_data, resample_pixel_spacing, resample_smoothing_beta, output_dir):
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

        logger.info("Hello, processing %s, %s", input_image_data, input_label_data)
        settings = Settings()

        print (settings)

        resample_pixel_spacing = np.full((3), resample_pixel_spacing)

        settings.img_interpolate.new_spacing = resample_pixel_spacing
        settings.roi_interpolate.new_spacing = resample_pixel_spacing
        settings.img_interpolate.smoothing_beta = resample_smoothing_beta

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
        properties = {
            "resampled_image_data": image_file,
            "resampled_label_data": label_file,
            "pertubation_set": f"{output_dir}/pertubations",
            "supervoxel_data": voxels_file
        }

        return properties
    

if __name__ == "__main__":
    cli()

