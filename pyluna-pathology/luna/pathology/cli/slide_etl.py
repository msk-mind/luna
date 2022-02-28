# General imports
import os
import logging
import click
import openslide

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

from dask.distributed import Client, as_completed
from luna.common.adapters import IOAdapter
from luna.common.utils import rebase_schema_numeric, apply_csv_filter, generate_uuid
from luna.pathology.common.utils import (
    get_downscaled_thumbnail,
    get_scale_factor_at_magnfication,
    get_stain_vectors_macenko,
)
from luna.common.custom_logger import init_logger
from luna.common.utils import cli_runner

import pyarrow.parquet as pq
from pyarrow import Table

init_logger()
logger = logging.getLogger("slide_etl")


_params_ = [
    ("input_slide_folder", str),
    ("comment", str),
    ("no_write", bool),
    ("subset_csv", str),
    ("debug_limit", int),
    ("num_cores", int),
    ("store_url", str),
    ("project_name", str),
    ("output_dir", str),
]

VALID_SLIDE_EXTENSIONS = [".svs", ".scn", ".tif"]


@click.command()
@click.argument("input_slide_folder", nargs=1)
@click.option(
    "-o",
    "--output_dir",
    required=False,
    help="path to output directory to save results",
)
@click.option(
    "-s",
    "--store_url",
    required=False,
    help="Where to store all slides in URL format (use file:///path/folder for local, s3://www.host:9000/ for s3)",
)
@click.option(
    "-p",
    "--project_name",
    required=False,
    help="project name to which slides are assigned or associated",
)
@click.option(
    "-c",
    "--comment",
    required=False,
    help="description/comments on the dataset (wrap in quotes)",
)
@click.option(
    "-sc",
    "--subset-csv",
    required=False,
    default="",
    help="path to a csv file with [string, include] schema to subset ingest data",
)
@click.option(
    "-dl",
    "--debug-limit",
    required=False,
    default=-1,
    help="limit number of slides process, for debugging, no_write is automatically enabled",
)
@click.option(
    "-nc", "--num_cores", required=False, help="Number of cores to use", default=4
)
@click.option("--no-write", is_flag=True, help="Disables write adapters")
@click.option(
    "-m",
    "--method_param_path",
    required=False,
    help="path to a metadata json/yaml file with method parameters to reproduce results",
)
def cli(**cli_kwargs):
    """Ingest slides by adding them to a file or s3 based storage location and generating metadata about them

    Output schema follows [ slide_id (str), slide_uuid (str), readable (bool), comment, project, ingest_time (UTC), generation_date (UTC), <store data | data_url >, <openslide data>, <stain data> ]

    \b
    Inputs:
        input_slide_folder: a slide image folder to walk over to find slide images

    \b
    Outputs:
        slide_table: a dataset representing the slide ingestion job

    \b
    Example:
        slide_etl /source/pathology/images
            --project_name PATH-123-45
            --comment "H&E needle core biospies"
            --store_url s3://localhost:9000/
            --num_cores 8
            -o /data/PATH-123-45/table
    """
    cli_runner(cli_kwargs, _params_, slide_etl)


def slide_etl(
    input_slide_folder,
    project_name,
    subset_csv,
    num_cores,
    no_write,
    debug_limit,
    store_url,
    comment,
    output_dir,
):
    """Ingest slides by adding them to a file or s3 based storage location and generating metadata about them

    Saves parquet table

    Args:
        input_slide_folder (str): path to parent directory containing slide images
        project_name (str): project name underwhich the slides should reside
        subset_csv (str): csv with filename and include/exclude criteria
            indicated by 1 or 0.
        comment (str): comment and description of dataset
        num_cores (int): number of cores (interfaced to dask)
        debug_limit (int): cap number fo slides to process below this number (for debugging, testing, no_write is automatically enabled)
        no_write (bool): don't actually generate or copy any data on disk, print outputs
        output_dir (str): path to output table


    Returns:
        dict: metadata about function call
    """
    slide_paths = []

    for path, _, files in os.walk(input_slide_folder):
        for file in files:
            file = os.path.join(path, file)

            if not Path(file).suffix in VALID_SLIDE_EXTENSIONS:
                continue

            slide_paths.append(file)

    slide_paths = apply_csv_filter(slide_paths, subset_csv)

    if debug_limit > 0:
        slide_paths = slide_paths[:debug_limit]

    if no_write:
        logger.info("Note, this is a dry run!!!")

    logger.info(f"Going to ingest {len(slide_paths)} slides!")

    write_adapter = IOAdapter(no_write=no_write).writer(store_url, bucket="pathology")

    sp = SlideProcessor(writer=write_adapter, project=project_name)

    with Client(
        host=os.environ["HOSTNAME"],
        n_workers=((num_cores + 1) // 2),
        threads_per_worker=2,
    ) as client:
        logger.info("Dashboard: " + client.dashboard_link)
        df = pd.DataFrame(
            [
                x.result()
                for x in tqdm(
                    as_completed(client.map(sp.run, slide_paths)),
                    smoothing=0.01,
                    total=len(slide_paths),
                )
            ]
        ).set_index("slide_id")

    df = df[df.index.notnull()]

    rebase_schema_numeric(df)

    df.loc[:, "project_name"] = project_name
    df.loc[:, "comment"] = comment
    df.loc[:, "generation_time"] = datetime.today()

    logger.info(df)

    output_table = os.path.join(output_dir, f"slide_ingest_{project_name}.parquet")


    pq.write_table( Table.from_pandas(df.drop(columns=['ingest_time', 'generation_time'])), output_table ) # Time columns causing issues in dremio

    logger.info(f"Saved table at {output_table}")

    properties = {"slide_table": output_table}

    return properties


class SlideProcessor:
    def __init__(self, writer, project):
        self.writer = writer
        self.project = project

    def check_slide(self, path) -> dict:
        try:
            openslide.OpenSlide(path)
            return {"slide_id": str(Path(path).stem), "valid_slide": True}
        except Exception:
            logger.warning(f"Couldn't open slide: {path}")
            return {"slide_id": np.nan, "valid_slide": False}

    def generate_properties(self, path) -> dict:
        try:
            slide = openslide.OpenSlide(path)
            kv = dict(slide.properties)
            kv["slide_uuid"] = generate_uuid(path, ["WSI"])
            return kv
        except Exception as err:
            logger.warning(f"Couldn't process slide: {path} - {err}")
            return {}

    def estimate_stain_type(self, path) -> dict:
        try:
            slide = openslide.OpenSlide(path)
            to_mag_scale_factor = get_scale_factor_at_magnfication(
                slide, requested_magnification=1
            )
            sample_arr = get_downscaled_thumbnail(slide, to_mag_scale_factor)
            stain_vectors = get_stain_vectors_macenko(sample_arr)
            return {
                "channel0_R": stain_vectors[0, 0],
                "channel0_G": stain_vectors[0, 1],
                "channel0_B": stain_vectors[0, 2],
                "channel1_R": stain_vectors[1, 0],
                "channel1_G": stain_vectors[1, 1],
                "channel1_B": stain_vectors[1, 2],
            }
        except Exception as err:
            logger.warning(f"Couldn't get stain vectors: {path} - {err}")
            return {}

    def run(self, path):
        """Extract openslide properties and write slide to storage location

        Args:
            path (string): path to slide image

        Returns:
            dict: slide metadata
        """

        kv = {}

        kv.update(self.check_slide(path))

        if not kv["valid_slide"]:
            return kv

        kv.update(self.generate_properties(path))
        kv.update(self.estimate_stain_type(path))
        kv.update(self.writer.write(path, f"{self.project}/slides"))

        return kv

    # Eventually it might be nice to automatically detect the stain type (at least H&E vs. DAB vs. Other)
    # def estimate_stain(self, slide):


if __name__ == "__main__":
    cli()
