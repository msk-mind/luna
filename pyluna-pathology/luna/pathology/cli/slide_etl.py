# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('slide_etl')

from luna.common.utils import cli_runner

_params_ = [('input_slide_folder', str), ('comment', str), ('slide_store_root', str), ('project_name', str), ('output_dir', str)]

VALID_SLIDE_EXTENSIONS = ['.svs', '.scn', '.tif']

@click.command()
@click.argument('input_slide_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-s', '--slide_store_root', required=False,
              help='Where to store all slides in URL format (use file:/path/folder for local, s3:/bucket/folder for s3)')
@click.option('-p', '--project_name', required=False,
              help='project name to which slides are assigned or associated')
@click.option('-c', '--comment', required=False,
              help='description/comments on the dataset (NB. wrap in quotes)')
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ A cli tool

    \b
    Inputs:
        input_slide_folder: a slide image folder
    \b
    Outputs:
        output data
    \b
    Example:
        CLI_TOOL ./slides/10001.svs ./halo/10001.job18484.annotations
            -an Tumor
            -o ./masks/10001/
    """
    cli_runner( cli_kwargs, _params_, slide_etl)


from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from urllib.parse import urlparse
import shutil
def slide_etl(input_slide_folder, project_name, slide_store_root, comment, output_dir):
    """ CLI tool method

    Args:
        input_data (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """

    url_result = urlparse(slide_store_root)

    if not url_result.scheme in ['file']:
        raise RuntimeError("Unsupported slide store schemes, please try s3:// or file://")
        
    if url_result.scheme == 'file':
        sp = SlideProcessor(slide_store_path = os.path.join (url_result.path, "pathology", project_name, 'slides'))

    slide_paths = []

    for path, _, files in os.walk(input_slide_folder):
        for file in files:
            file = os.path.join(path, file)

            if not Path(file).suffix in VALID_SLIDE_EXTENSIONS: continue

            slide_paths.append(file)


    with ProcessPoolExecutor(4) as executor:
        df = pd.DataFrame(tqdm(executor.map(sp.run, slide_paths[:16]), total=len(slide_paths))).set_index('slide_id')

    df.loc[:, 'project_name'] = project_name
    df.loc[:, 'comment'] = comment

    output_table = os.path.join(output_dir, "slide_ingest.csv")

    df.to_csv(output_table)

    properties = {
        'slide_table': output_table
    }

    return properties


import openslide
from luna.pathology.common.utils import get_downscaled_thumbnail, get_scale_factor_at_magnfication, get_stain_vectors_macenko, pull_stain_channel, read_tile_bytes
from luna.common.utils import generate_uuid
import numpy as np

class SlideProcessor:
    def __init__(self, slide_store_path):
        self.slide_store_path = slide_store_path

        os.makedirs(slide_store_path, exist_ok=True)

        logger.info ("Writing slides to : " + slide_store_path)

    def run(self, path):
        """Extract openslide properties

        Args:
            path (string): path to slide image

        Returns:
            dict: slide metadata
        """
        slide =  openslide.OpenSlide(path)

        input_slide_path  = Path(path)
        output_slide_path = os.path.join(self.slide_store_path, input_slide_path.name)
        
        kv = dict(slide.properties)

        kv['slide_id']   = input_slide_path.stem
        kv['slide_uuid'] = ''#generate_uuid(path, ['WSI'])

        os.path.join(self.slide_store_path, )

        shutil.copyfile(input_slide_path, output_slide_path)

        kv['slide_path'] = output_slide_path


        return kv

    def estimate_stain(self, slide):

        to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=1)
        sample_arr    = get_downscaled_thumbnail(slide, to_mag_scale_factor)
        stain_vectors = get_stain_vectors_macenko(sample_arr)

        if stain_vectors[1, 0] < 0.5 * stain_vectors[1, 1]:
            return 'H&E'
        else:
            return 'Other'

if __name__ == "__main__":
    cli()




self._client = Minio(params['MINIO_URI'], access_key=params['MINIO_USER'], secret_key=params['MINIO_PASSWORD'], secure=False)

if self.params.get('OBJECT_STORE_ENABLED',  False):
if not self._client.bucket_exists(self._bucket_id):
self._client.make_bucket(self._bucket_id)
future = executor.submit(self._client.fput_object, object_bucket, f"{object_folder}/{p.name}", p, part_size=250000000)
