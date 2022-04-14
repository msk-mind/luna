import click
import os
import logging
import shutil

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger
import luna.common.constants as const

from luna.pathology.common.annotation_utils import (
    convert_slide_bitmap_to_geojson,
    check_slideviewer_and_download_bmp,
)
from luna.pathology.common.slideviewer_client import fetch_slide_ids

import pandas as pd
import numpy as np

from dask.distributed import Client, as_completed


@click.command()
@click.option(
    "-d",
    "--data_config_file",
    default=None,
    type=click.Path(exists=True),
    help="path to yaml file containing data input and output parameters. "
    "See dask_data_config.yaml.template",
)
@click.option(
    "-a",
    "--app_config_file",
    default="config.yaml",
    type=click.Path(exists=True),
    help="path to yaml file containing application runtime parameters. "
    "See config.yaml.template",
)
def cli(data_config_file, app_config_file):
    """This module generates parquets with regional annotation pathology data

    INPUT PARAMETERS

    app_config_file - path to yaml file containing application runtime parameters. See config.yaml.template

    data_config_file - path to yaml file containing data input and output parameters. See dask_data_config.yaml.template

    TABLE SCHEMA

    - sv_project_id: project number in slide viewer

    - slideviewer_path: slide path based on slideviewer organization

    - slide_id: slide id. synonymous with image_id

    - user: username of the annotator for a given annotation. For all slides, we combine multiple annotations from
        different users for a slide. In this case, user is set to 'CONCAT' and bmp_filepath, npy_filepath are null.

    - bmp_filepath: file path to downloaded bmp annotation file

    - npy_filepath: file path to npy annotation file converted from bmp

    - geojson_path: file path to  geojson file converted from numpy

    - date: creation date

    - labelset:
    """
    logger = init_logger()

    # load configs
    cfg = ConfigSet(name="DATA_CFG", config_file=data_config_file)
    cfg = ConfigSet(name="APP_CFG", config_file=app_config_file)

    with CodeTimer(logger, "generate annotation geojson table"):
        logger.info("data template: " + data_config_file)
        logger.info("config_file: " + app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        failed = create_geojson_table()

        if failed:
            logger.error("GEOJSON table creation had errors. Exiting.")
            logger.error(failed)
            raise RuntimeError("GEOJSON table creation had errors. Exiting.")

        return


def create_geojson_table():
    """Vectorizes npy array annotation file into polygons and builds GeoJson with the polygon features.
    Creates a geojson file per labelset.
    Combines multiple annotations from different users for a slide.

    Returns:
        list: list of slide ids that failed
    """
    logger = logging.getLogger(__name__)

    failed = []
    # get application and data config variables
    cfg = ConfigSet()
    client = Client(n_workers=25, threads_per_worker=1, memory_limit=0.1)
    client.run(init_logger)
    logger.info(client)

    SLIDEVIEWER_API_URL = cfg.get_value("DATA_CFG::SLIDEVIEWER_API_URL")
    SLIDEVIEWER_CSV_FILE = cfg.get_value("DATA_CFG::SLIDEVIEWER_CSV_FILE")
    PROJECT_ID = cfg.get_value("DATA_CFG::PROJECT_ID")
    LANDING_PATH = cfg.get_value("DATA_CFG::LANDING_PATH")
    TMP_ZIP_DIR_NAME = cfg.get_value("DATA_CFG::REQUESTOR_DEPARTMENT") + "_tmp_zips"
    TMP_ZIP_DIR = os.path.join(LANDING_PATH, TMP_ZIP_DIR_NAME)
    SLIDE_BMP_DIR = os.path.join(LANDING_PATH, "regional_bmps")
    SLIDE_NPY_DIR = os.path.join(LANDING_PATH, "regional_npys")
    SLIDE_STORE_DIR = os.path.join(LANDING_PATH, "slides")
    TABLE_OUT_DIR = const.TABLE_LOCATION(cfg)

    os.makedirs(TABLE_OUT_DIR, exist_ok=True)
    logger.info("Table output directory = %s", TABLE_OUT_DIR)

    # setup variables needed for build geojson UDF
    contour_level = cfg.get_value("DATA_CFG::CONTOUR_LEVEL")

    # fetch full set of slideviewer slides for project
    slides = fetch_slide_ids(
        SLIDEVIEWER_API_URL,
        PROJECT_ID,
        const.CONFIG_LOCATION(cfg),
        SLIDEVIEWER_CSV_FILE,
    )
    df = pd.DataFrame(
        data=np.array(slides), columns=["slideviewer_path", "slide_id", "sv_project_id"]
    )

    # get users and labelsets for df explosion
    all_users_list = cfg.get_value("DATA_CFG::USERS")
    all_labelsets = cfg.get_value("DATA_CFG::LABEL_SETS")

    global params
    params = cfg.get_config_set("APP_CFG")

    bmp_jobs = []
    for _, row in df.iterrows():
        bmp_future = client.submit(
            check_slideviewer_and_download_bmp,
            row.sv_project_id,
            row.slideviewer_path,
            row.slide_id,
            all_users_list,
            SLIDE_BMP_DIR,
            SLIDEVIEWER_API_URL,
            TMP_ZIP_DIR,
        )
        bmp_jobs.append(bmp_future)

    json_jobs = []
    for bmp_future in as_completed(bmp_jobs):
        if bmp_future.result() is not None:
            json_future = client.submit(
                convert_slide_bitmap_to_geojson,
                bmp_future,
                all_labelsets,
                contour_level,
                SLIDE_NPY_DIR,
                SLIDE_STORE_DIR,
            )
            json_jobs.append(json_future)

    for json_future in as_completed(json_jobs):
        slide_id = -1
        try:
            if json_future.result() is not None:
                slide_id, data = json_future.result()
                if slide_id and data:
                    result_df = pd.DataFrame(data)
                    logger.info(result_df)
                    result_df.drop(columns="geojson").to_parquet(
                        f"{TABLE_OUT_DIR}/regional_annot_slice_slide={slide_id}.parquet"
                    )
                else:
                    failed.append(slide_id)
                    logger.warning(
                        "Empty geojson returned. this means either this was an empty slide or an error occured during geojson generate"
                    )
        except Exception:
            failed.append(slide_id)
            logger.warning(f"Something was wrong with future {json_future}, skipping.")

    client.shutdown()

    return failed


if __name__ == "__main__":
    cli()
