import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('coregister_volumes')

from luna.common.utils import cli_runner

_params_ = [('input_image_1', str), ('input_image_2', str), ('output_dir', str), ('resample_pixel_spacing', float), ('save_npy', bool)]

@click.command()
@click.option('-i1', '--input_image_1', required=False,
              help='path to input data')
@click.option('-i2', '--input_image_2', required=False,
              help='path to output directory to save results')
@click.option('-npy', '--save_npy', required=False, is_flag=True,
              help='path to output directory to save results')
@click.option('-rps', '--resample_pixel_spacing', required=False,
              help='path to input label data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """
    Run with explicit arguments:

    \b
        infer_tiles
            -i 1412934/data/TileImages
            -o 1412934/data/TilePredictions
            -rn msk-mind/luna-ml:main 
            -tn tissue_tile_net_transform 
            -mn tissue_tile_net_model_5_class
            -wt main:tissue_net_2021-01-19_21.05.24-e17.pth

    Run with implicit arguments:

    \b
        infer_tiles -m 1412934/data/TilePredictions/metadata.json
    
    Run with mixed arguments (CLI args override yaml/json arguments):

    \b
        infer_tiles --input_data 1412934/data/TileImages -m 1412934/data/TilePredictions/metadata.json
    """
    cli_runner(cli_kwargs, _params_, coregister_volumes)


import numpy as np
import scipy.ndimage
from luna.radiology.mirp.imageReaders import read_itk_image, read_itk_segmentation
from pathlib import Path

def coregister_volumes(input_image_1, input_image_2, resample_pixel_spacing, output_dir, save_npy):
    resample_pixel_spacing = np.full((3), resample_pixel_spacing)

    image_class_object_1      = read_itk_image(input_image_1, modality=str(Path(input_image_1).stem))
    image_class_object_2      = read_itk_image(input_image_2, modality=str(Path(input_image_2).stem))

    ct_iso = interpolate(image_class_object_1, resample_pixel_spacing, reference_geometry=image_class_object_1)
    pt_iso = interpolate(image_class_object_2, resample_pixel_spacing, reference_geometry=image_class_object_1)

    image_file_1 = image_class_object_1.export(file_path=output_dir)
    image_file_2 = image_class_object_2.export(file_path=output_dir)

    if save_npy:
        np.save(image_file_1 + '.npy', image_class_object_1.get_voxel_grid())
        np.save(image_file_2 + '.npy', image_class_object_2.get_voxel_grid())

    return {
        'reference_origin': list(image_class_object_1.origin)
    }


def interpolate(image, resample_spacing, reference_geometry):

    image_size    = image.size
    image_spacing = image.spacing
    image_voxels  = image.voxel_grid
    image_origin  = image.origin
    print ("Image size, spacing=", image_size, image_spacing)

    reference_origin = reference_geometry.origin
    
    reference_offset = (reference_origin - image_origin) / image_spacing
    print ("Reference offset=", reference_offset)

    grid_spacing = resample_spacing / image_spacing
    
    grid_size = np.ceil(np.multiply(reference_geometry.size, reference_geometry.spacing / resample_spacing))
    
    print ("Grid spacing, size=", grid_spacing, grid_size)
    
    map_z, map_y, map_x = np.mgrid[:grid_size[0], :grid_size[1], :grid_size[2]]
    
    map_z = map_z * grid_spacing[0] + reference_offset[0]
    map_y = map_y * grid_spacing[1] + reference_offset[1]
    map_x = map_x * grid_spacing[2] + reference_offset[2]
    
    print ("Z, Y, Z=", map_z.shape, map_y.shape, map_x.shape)

    resampled_image = scipy.ndimage.map_coordinates(input=image_voxels.astype(np.float32),
                                      coordinates=np.array([map_z, map_y, map_x], dtype=np.float32), order=3, mode='nearest') 
    print ("Resampled size=", resampled_image.shape)
    
    image.set_spacing (resample_spacing)
    image.set_origin (image_origin + (reference_origin - image_origin))
    image.set_voxel_grid(voxel_grid=resampled_image)
    
    print ("New origin=", image.origin)
    
    return resampled_image 

if __name__ == "__main__":
    cli()