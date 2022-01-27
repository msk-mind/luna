# General imports
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('slide_etl')

from luna.common.utils import cli_runner

_params_ = [('input_slide_folder', str), ('comment', str), ('dry_run', bool), ('num_cores', int), ('slide_store_root', str), ('project_name', str), ('output_dir', str)]

VALID_SLIDE_EXTENSIONS = ['.svs', '.scn', '.tif']

@click.command()
@click.argument('input_slide_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-s', '--slide_store_root', required=False,
              help='Where to store all slides in URL format (use file:///path/folder for local, s3://www.host:9000/ for s3)')
@click.option('-p', '--project_name', required=False,
              help='project name to which slides are assigned or associated')
@click.option('-c', '--comment', required=False,
              help='description/comments on the dataset (NB. wrap in quotes)')
@click.option('-nc', '--num_cores', required=False,
              help='Number of cores to use', default=4)
@click.option('--dry-run', is_flag=True,
              help="Only print data, no data is copied or generated on disk.")
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


def rebase_schema_numeric(df):
    for col in df.columns:
        df[col] = df[col].astype(float, errors='ignore')
        df[col] = df[col].astype(int,   errors='ignore')

from pathlib import Path
import pandas as pd
from urllib.parse import urlparse
import shutil

from distributed import Client

def slide_etl(input_slide_folder, project_name, num_cores, dry_run, slide_store_root, comment, output_dir):
    """ CLI tool method

    Args:
        input_data (str): path to input data
        output_dir (str): output/working directory

    Returns:
        dict: metadata about function call
    """

    url_result = urlparse(slide_store_root)

    if not url_result.scheme in ['file', 's3']:
        raise RuntimeError("Unsupported slide store schemes, please try s3:// or file://")
    
    relative_path = os.path.join("pathology", project_name, 'slides')

    if url_result.scheme == 'file':
        adapter = FileWriteAdatper(store_path=os.path.join(url_result.path, relative_path))

    if url_result.scheme == 's3':
        adapter = MinioWriteAdatper(store_hostname=url_result.hostname, store_port=url_result.port, bucket='pathology', prefix=os.path.join(project_name, 'slides'))
        
    slide_paths = []

    for path, _, files in os.walk(input_slide_folder):
        for file in files:
            file = os.path.join(path, file)

            if not Path(file).suffix in VALID_SLIDE_EXTENSIONS: continue

            slide_paths.append(file)

    sp = SlideProcessor(data_store_adapter=adapter)

    with Client(host=os.environ['HOSTNAME'], n_workers=((num_cores + 3) // 4), threads_per_worker=4) as client:
        logger.info (client.dashboard_link)
        df = pd.DataFrame(client.gather(client.map(sp.run, slide_paths[:]))).set_index('slide_id')

    df = df[df.index.notnull()]

    rebase_schema_numeric(df)

    df.loc[:, 'project_name'] = project_name
    df.loc[:, 'comment'] = comment

    print (df)

    if dry_run:
        return {}

    output_table = os.path.join(output_dir, "slide_ingest.parquet")
    df.to_parquet(output_table)

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

class FileWriteAdatper:
    def __init__(self, store_path):
        self.store_path = store_path
        print("Writing to: " + self.store_path)

        os.makedirs(self.store_path, exist_ok=True)

        self.base_url = f'file://{Path(store_path)}'
        print("Base URL: " + self.base_url)

    def write(self, input_data):
        input_data  = Path(input_data)
        filename = input_data.name

        shutil.copy(input_data, self.store_path)

        output_data_url = os.path.join(self.base_url, filename)

        return {'data_url': output_data_url}

from minio import Minio
class MinioWriteAdatper:
    def __init__(self, store_hostname, store_port, bucket, prefix):
        self.store_hostname = store_hostname
        self.store_port = store_port
        self.bucket = bucket
        self.prefix = prefix

        print (self.store_hostname, self.store_port, self.bucket, self.prefix )

        client = Minio(f'{self.store_hostname}:{self.store_port}', access_key=os.environ['MINIO_USER'], secret_key=os.environ['MINIO_PASSWORD'], secure=True)

        if not client.bucket_exists(self.bucket):
            client.make_bucket(self.bucket)
       

    def write(self, input_data):

        input_data  = Path(input_data)
        filename = input_data.name

        client = Minio(f'{self.store_hostname}:{self.store_port}', access_key=os.environ['MINIO_USER'], secret_key=os.environ['MINIO_PASSWORD'], secure=True)
        client.fput_object(self.bucket, f"{self.prefix}/{filename}", input_data, part_size=250000000)

        output_data_url = os.path.join(f"s3://{self.store_hostname}:{self.store_port}", self.bucket, f"{self.prefix}/{filename}")

        return {'data_url': output_data_url}

class ReadAdapter:
    def __init__(self): pass

    def stat(self, url):
        url_result = urlparse(url)

        if not url_result.scheme in ['file', 's3']:
            raise RuntimeWarning("Unsupported slide store schemes, please try s3:// or file://, skipping...")

        if url_result.scheme == 'file':
            print (os.stat(url_result.path))
    

class SlideProcessor:
    def __init__(self, data_store_adapter):
        self.data_store_adapter = data_store_adapter

    def run(self, path):
        """Extract openslide properties

        Args:
            path (string): path to slide image

        Returns:
            dict: slide metadata
        """
        try:
            slide =  openslide.OpenSlide(path)

        except:
            print ("Couldn't process slide: ", path)
            return {'slide_id':np.nan}
        
        else:

            input_slide_path  = Path(path)
            
            kv = dict(slide.properties)

            kv['slide_id']   = str(input_slide_path.stem)
            kv['slide_uuid'] = ''#generate_uuid(path, ['WSI'])

            write_kv = self.data_store_adapter.write(path)

            kv.update(write_kv)

            slide.close()

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


# self._client = Minio(params['MINIO_URI'], access_key=params['MINIO_USER'], secret_key=params['MINIO_PASSWORD'], secure=False)

# if self.params.get('OBJECT_STORE_ENABLED',  False):
# if not self._client.bucket_exists(self._bucket_id):
# self._client.make_bucket(self._bucket_id)
# future = executor.submit(self._client.fput_object, object_bucket, f"{object_folder}/{p.name}", p, part_size=250000000)
