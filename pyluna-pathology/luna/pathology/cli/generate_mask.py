# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_mask')

from luna.common.utils import cli_runner

_params_ = [('input_slide_image', str), ('input_slide_roi', str), ('output_dir', str), ('annotation_name', str)]

@click.command()
@click.argument('input_slide_image', nargs=1)
@click.argument('input_slide_roi', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-an', '--annotation_name', required=False,
              help="annotation layer name to use (e.g. Tumor, Tissue)")
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Generate a full resolution mask image (.tif) from vector annotations (polygons, shapes)

    \b
    Inputs:
        input_slide_image: slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_roi: roi containing vector shapes (*.annotations, *.json)
    \b
    Outputs:
        slide_mask
    \b
    Example:
        generate_mask ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    cli_runner( cli_kwargs, _params_, generate_mask)


import openslide
import tifffile
import numpy as np
from PIL import Image
from luna.pathology.common.utils import get_layer_names, convert_xml_to_mask
from skimage.measure import block_reduce
def generate_mask(input_slide_image, input_slide_roi, output_dir, annotation_name):
    """ Generate a full resolution mask image (.tif) from vector annotations (polygons, shapes)

    Take into account positive and negative spaces.  Essentially rasterizes a polygon file.

    Args:
        input_slide_image (str): path to slide image (virtual slide formats compatible with openslide, .svs, .tif, .scn, ...)
        input_slide_roi (str): path to a halo or other polygonal annotation file (.xml, .geojson)
        output_dir (str): output/working directory
        annotation_name (str): name of annotation layer to use

    Returns:
        dict: metadata about function call
    """

    slide = openslide.OpenSlide(input_slide_image)
    slide.get_thumbnail((1000, 1000)).save(f"{output_dir}/slide_thumbnail.png")

    wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions
    logger.info(f"Slide shape={wsi_shape}")

    layer_names     = get_layer_names(input_slide_roi)
    logger.info(f"Available layer names={layer_names}")

    mask_arr = convert_xml_to_mask(input_slide_roi, wsi_shape, annotation_name)

    logger.info(f"Generating mask thumbnail, mask size={mask_arr.shape}")
    openslide.ImageSlide(Image.fromarray(255 * block_reduce(mask_arr, block_size=(10, 10), func=np.mean, cval=0.0))).get_thumbnail((1000, 1000)).save(f"{output_dir}/mask_thumbnail.png")

    slide_mask = f"{output_dir}/mask_full_res.tif"
    tifffile.imsave(slide_mask, mask_arr, compress=5)

    properties = {
        'slide_mask': slide_mask,
        'mask_size': list (wsi_shape),
    }

    return properties

if __name__ == "__main__":
    cli()
