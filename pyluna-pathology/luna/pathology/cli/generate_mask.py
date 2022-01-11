# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_mask')

from luna.common.utils import cli_runner

_params_ = [('input_slide_path', str), ('input_roi_path', str), ('output_dir', str), ('annotation_name', str)]

@click.command()
@click.option('-insp', '--input_slide_path', required=False,
              help='path to input data')
@click.option('-inrp', '--input_roi_path', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-an', '--annotation_name', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """ 

    """
    cli_runner( cli_kwargs, _params_, generate_mask)


import openslide
import tifffile
import numpy as np
from PIL import Image
from luna.pathology.common.utils import get_layer_names, convert_xml_to_mask
from skimage.measure import block_reduce

def generate_mask(input_slide_path, input_roi_path, output_dir, annotation_name):
    slide = openslide.OpenSlide(input_slide_path)
    slide.get_thumbnail((1000, 1000)).save(f"{output_dir}/slide_thumbnail.png")

    wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions
    logger.info(f"Slide shape={wsi_shape}")

    layer_names     = get_layer_names(input_roi_path)
    logger.info(f"Available layer names={layer_names}")

    mask_arr = convert_xml_to_mask(input_roi_path, wsi_shape, annotation_name)

    logger.info(f"Generating mask thumbnail, mask size={mask_arr.shape}")
    openslide.ImageSlide(Image.fromarray(255 * block_reduce(mask_arr, block_size=(10, 10), func=np.mean, cval=0.0))).get_thumbnail((1000, 1000)).save(f"{output_dir}/mask_thumbnail.png")

    tifffile.imsave(f"{output_dir}/mask_full_res.tif", mask_arr, compress=5)

    properties = {
        'mask_size': wsi_shape,
        'data': f"{output_dir}/mask_full_res.tif"
    }

    return properties

if __name__ == "__main__":
    cli()
