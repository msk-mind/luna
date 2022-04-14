import logging
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('coregister_volumes')

from luna.common.utils import cli_runner

_params_ = [('input_itk_volume', str), ('input_itk_geometry', str), ('output_dir', str), ('order', int), ('resample_pixel_spacing', float), ('save_npy', bool)]

@click.command()
@click.argument('input_itk_volume', nargs=1)
@click.argument('input_itk_geometry', nargs=1)
@click.option('-npy', '--save_npy', required=False, is_flag=True,
              help='whether to additionally save the volumes as numpy files')
@click.option('-rps', '--resample_pixel_spacing', required=False,
              help='isotropic voxel size (in mm)')
@click.option('-ord', '--order', required=False,
              help='interpolation order')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """Resamples and co-registeres all volumes to occupy the same physical coordinates of a reference geometry (given as a itk_volume)

    NB: Pass the same image as both the volume and geometry to simply resample a given volume. Useful for PET/CT coregistration.

    \b
    Inputs:
        input_itk_volume: itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        input_itk_geometry: itk compatible image volume (.mhd, .nrrd, .nii, etc.) to use as reference geometry
    \b
    Outputs:
        itk_volume
    \b
    Example: (Resample a pet image to match dimensions of a CT image)
        coregister_volumes ./scans/10001/CT/pet_image.nii ./scans/10001/CT/ct_image.nii
            --resample_pixel_spacing 1.5
            --order 3
            --save_npy
            -o ./registered/
    """
    cli_runner(cli_kwargs, _params_, coregister_volumes)


import numpy as np
import scipy.ndimage
from luna.radiology.mirp.imageReaders import read_itk_image, read_itk_segmentation
from pathlib import Path

def coregister_volumes(input_itk_volume: str, input_itk_geometry: str, resample_pixel_spacing: float, output_dir: str, order: int, save_npy: bool):
    """Resamples and co-registeres all volumes to occupy the same physical coordinates of a reference geometry (given as a itk_volume) and desired voxel size

    Args:
        input_itk_volume (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.)
        input_itk_geometry (str): path to itk compatible image volume (.mhd, .nrrd, .nii, etc.) to use as a reference geometry
        output_dir (str): output/working directory
        resample_pixel_spacing (float): voxel size in mm
        order (int): interpolation order [0-5]
        save_npy(bool): whether to also save a numpy file representing the volume

    Returns:
        dict: metadata about function call
    """
    resample_pixel_spacing = np.full((3), resample_pixel_spacing)

    image_class_object_volume      = read_itk_image(input_itk_volume, modality=str(Path(input_itk_volume).stem))
    image_class_object_geometry    = read_itk_image(input_itk_geometry, modality=str(Path(input_itk_geometry).stem))

    voxels_iso = interpolate(image_class_object_volume, resample_pixel_spacing, reference_geometry=image_class_object_geometry, order=order)

    image_file = image_class_object_volume.export(file_path=output_dir)
    
    if save_npy:
        np.save(image_file + '.npy', image_class_object_volume.get_voxel_grid())

    return {
        'itk_volume': image_file,
        'npy_volume': image_file + '.npy',
    }


def interpolate(image, resample_spacing, reference_geometry, order=3):
    """Run interplation 
    
    Args:
        image (imageClass): mirp image class object
        resample_spacing (np.ndarray): spacing of resample as a 1D vector
        reference_geometry (imageClass): output/working directory
        order (int): interpolation order [0-5]

    Returns:
        imageClass: mirp image class object after resample
    """

    image_size    = image.size
    image_spacing = image.spacing
    image_voxels  = image.voxel_grid
    image_origin  = image.origin
    logger.info ("Image size=%s, spacing=%s" % (image_size, image_spacing))

    reference_origin = reference_geometry.origin
    
    reference_offset = (reference_origin - image_origin) / image_spacing
    logger.info ("Reference offset=%s" % (reference_offset))

    grid_spacing = resample_spacing / image_spacing
    
    grid_size = np.ceil(np.multiply(reference_geometry.size, reference_geometry.spacing / resample_spacing))
    
    logger.info ("Grid spacing=%s, size=%s" % (grid_spacing, grid_size))
    
    map_z, map_y, map_x = np.mgrid[:grid_size[0], :grid_size[1], :grid_size[2]]
    
    map_z = map_z * grid_spacing[0] + reference_offset[0]
    map_y = map_y * grid_spacing[1] + reference_offset[1]
    map_x = map_x * grid_spacing[2] + reference_offset[2]
    
    logger.info ("Z=%s, Y=%s, Z=%s" % (map_z.shape, map_y.shape, map_x.shape))

    resampled_image = scipy.ndimage.map_coordinates(input=image_voxels.astype(np.float32),
                                      coordinates=np.array([map_z, map_y, map_x], dtype=np.float32), 
                                      order=order, mode='nearest') 

    print (resampled_image.shape)
    logger.info ("Resampled size=%s" % (list(resampled_image.shape)))
    
    image.set_spacing (resample_spacing)
    image.set_origin (image_origin + (reference_origin - image_origin))
    image.set_voxel_grid(voxel_grid=resampled_image)
    
    logger.info ("New origin=%s" % (image.origin))
    
    return resampled_image 

if __name__ == "__main__":
    cli()
