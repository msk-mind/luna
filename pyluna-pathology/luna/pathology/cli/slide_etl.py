# General imports
from distutils.log import debug
import os, json, logging, yaml
import click

from luna.common.custom_logger   import init_logger

init_logger()
logger = logging.getLogger('slide_etl')

from luna.common.utils import cli_runner

_params_ = [('input_slide_folder', str), ('comment', str), ('dry_run', bool), ('debug_limit', int), ('num_cores', int), ('store_url', str), ('project_name', str), ('output_dir', str)]

VALID_SLIDE_EXTENSIONS = ['.svs', '.scn', '.tif']

@click.command()
@click.argument('input_slide_folder', nargs=1)
@click.option('-o', '--output_dir', required=False,
              help='path to output directory to save results')
@click.option('-s', '--store_url', required=False,
              help='Where to store all slides in URL format (use file:///path/folder for local, s3://www.host:9000/ for s3)')
@click.option('-p', '--project_name', required=False,
              help='project name to which slides are assigned or associated')
@click.option('-c', '--comment', required=False,
              help='description/comments on the dataset (wrap in quotes)')
@click.option('-dl', '--debug-limit', required=False, default=-1,
              help='limit number of slides process, for debugging, dry_run is automatically enabled')
@click.option('-nc', '--num_cores', required=False,
              help='Number of cores to use', default=4)
@click.option('--dry-run', is_flag=True,
              help="Only print data, no data is copied or generated on disk.")
@click.option('-m', '--method_param_path', required=False,
              help='path to a metadata json/yaml file with method parameters to reproduce results')
def cli(**cli_kwargs):
    """ Ingest slides by adding them to a file or s3 based storage location and generating metadata about them

    \b
    Inputs:
        input_slide_folder: a slide image folder to walk over to find slide images
    \b
    Outputs:
        slide_table: a dataset representing the slide ingestion job
    \b
    Example:
        slide_etl /source/pathology/images \
            --project_name PATH-123-45 \
            --comment "H&E needle core biospies" \
            --store_url s3://localhost:9000/ \
            --num_cores 8 \
            -o /data/PATH-123-45/table
    """
    cli_runner( cli_kwargs, _params_, slide_etl)


from pathlib import Path
import pandas as pd
from tqdm import tqdm

from dask.distributed import Client, as_completed
from luna.common.adapters import IOAdapter
from luna.common.utils import rebase_schema_numeric


import openslide
from luna.common.utils import generate_uuid
import numpy as np

from datetime import datetime

def slide_etl(input_slide_folder, project_name, num_cores, dry_run, debug_limit, store_url, comment, output_dir):
    """ CLI tool method

    Args:
        input_slide_folder (str): path to parent directory containing slide images
        project_name (str): project name underwhich the slides should reside
        comment (str): comment and description of dataset
        num_cores (int): number of cores (interfaced to dask)
        debug_limit (int): cap number fo slides to process below this number (for debugging, testing, dry_run is automatically enabled)
        dry_run (bool): don't actually generate or copy any data on disk, print outputs
        output_dir (str): path to output table


    Returns:
        dict: metadata about function call
    """
    slide_paths = []

    for path, _, files in os.walk(input_slide_folder):
        for file in files:
            file = os.path.join(path, file)

            if not Path(file).suffix in VALID_SLIDE_EXTENSIONS: continue

            slide_paths.append(file)


    if debug_limit > 0: 
        slide_paths = slide_paths[:debug_limit]
        dry_run = True

    if dry_run: logger.info ("Note, this is a dry run!!!")

    logger.info(slide_paths)

    write_adapter = IOAdapter(dry_run=dry_run).writer(store_url, bucket='pathology')

    sp = SlideProcessor(writer=write_adapter, project=project_name)

    with Client(host=os.environ['HOSTNAME'], n_workers=((num_cores + 1) // 2), threads_per_worker=2) as client:
        logger.info ("Dashboard: "+ client.dashboard_link)
        df = pd.DataFrame([x.result() for x in tqdm(as_completed(client.map(sp.run, slide_paths)), smoothing=0.01, total=len(slide_paths))]).set_index('slide_id')

    df = df[df.index.notnull()]

    rebase_schema_numeric(df)

    df.loc[:, 'project_name'] = project_name
    df.loc[:, 'comment'] = comment
    df.loc[:, 'ingestion_date'] = datetime.today()

    logger.info(df)

    if dry_run:
        return {}

    output_table = os.path.join(output_dir, f"slide_ingest_{project_name}.parquet")
    df.to_parquet(output_table)

    properties = {
        'slide_table': output_table
    }

    return properties


class SlideProcessor:
    def __init__(self, writer, project):
        self.writer  = writer
        self.project = project

    def run(self, path):
        """Extract openslide properties and write slide to storage location

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
            kv['slide_uuid'] = generate_uuid(path, ['WSI'])

            write_kv = self.writer.write(path, f'{self.project}/slides')

            kv.update(write_kv)

            slide.close()

            return kv

    # Eventually it might be nice to automatically detect the stain type (at least H&E vs. DAB vs. Other)
    # def estimate_stain(self, slide):
    #     from luna.pathology.common.utils import get_downscaled_thumbnail, get_scale_factor_at_magnfication, get_stain_vectors_macenko, pull_stain_channel, read_tile_bytes

    #     to_mag_scale_factor = get_scale_factor_at_magnfication (slide, requested_magnification=1)
    #     sample_arr    = get_downscaled_thumbnail(slide, to_mag_scale_factor)
    #     stain_vectors = get_stain_vectors_macenko(sample_arr)


    #     if stain_vectors[1, 0] < 0.5 * stain_vectors[1, 1]:
    #         return 'H&E'
    #     else:
    #         return 'Other'

if __name__ == "__main__":
    cli()