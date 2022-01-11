# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('generate_mask')

from luna.common.utils import cli_runner

_params_ = [('input_data', str), ('output_dir', str), ('repo_name', str), ('transform_name', str), ('model_name', str), ('weight_tag', str), ('num_cores', int), ('batch_size', int)]

@click.command()
@click.option('-i', '--input_data', required=False,
              help='path to input data')
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-rn', '--repo_name', required=False,
              help="repository name to pull model and weight from, e.g. msk-mind/luna-ml")
@click.option('-tn', '--transform_name', required=False,
              help="torch hub transform name")   
@click.option('-mn', '--model_name', required=False,
              help="torch hub model name")    
@click.option('-wt', '--weight_tag', required=False,
              help="weight tag filename")  
@click.option('-nc', '--num_cores', required=False,
              help="Number of cores to use", default=4)  
@click.option('-bx', '--batch_size', required=False,
              help="weight tag filename", default=256)    
@click.option('-m', '--method_param_path', required=False,
              help='json file with method parameters for tile generation and filtering')
def cli(**cli_kwargs):
    """ 

    """
    cli_runner( cli_kwargs, _params_, generate_mask)


def generate_mask():
    slide = openslide.OpenSlide(slide_path)
    slide.get_thumbnail((1000, 1000)).save(f"{output_dir}/slide_thumbnail.png")

    wsi_shape = slide.dimensions[1], slide.dimensions[0] # Annotation file has flipped dimensions w.r.t openslide conventions
    self.logger.info(f"Slide shape={wsi_shape}")

    layer_names     = get_layer_names(roi_path)
    self.logger.info(f"Available layer names={layer_names}")
    
    # x_pol, y_pol    = get_polygon_bounding_box(roi_path, self.annotation_name)
    # self.logger.info(f"Bounding box={x_pol}, {y_pol}")

    # x_roi, y_roi    = convert_halo_xml_to_roi(roi_path)

    mask_arr = convert_xml_to_mask(roi_path, wsi_shape, self.annotation_name)

#        openslide.ImageSlide(Image.fromarray(255 * mask_arr)).get_thumbnail((1000, 1000)).save(f"{output_dir}/mask_thumbnail.png")

    tifffile.imsave(f"{output_dir}/mask_full_res.tif", mask_arr, compress=5)

    properties = {
        'mask_size': wsi_shape,
        'data': f"{output_dir}/mask_full_res.tif"
    }

    self.logger.info(properties)

    return properties