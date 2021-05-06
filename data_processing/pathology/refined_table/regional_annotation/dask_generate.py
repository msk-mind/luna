import click
import os

from data_processing.common.CodeTimer       import CodeTimer
from data_processing.common.config          import ConfigSet
from data_processing.common.custom_logger   import init_logger
import data_processing.common.constants as const

from data_processing.pathology.common.annotation_utils   import convert_slide_bitmap_to_geojson, check_slideviewer_and_download_bmp
from data_processing.pathology.common.slideviewer_client import fetch_slide_ids

import pandas as pd
import numpy as np

from dask.distributed import Client, as_completed

logger = init_logger()

@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
@click.option('-p', '--process_string', default='geojson',
              help='process to run or replay: e.g. geojson OR concat')
def cli(data_config_file, app_config_file, process_string):
    """
        This module generates a delta table with geojson pathology data based on the input and output parameters
         specified in the data_config_file.

        Example:
            python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file> \
                     --process_string geojson
    """
    with CodeTimer(logger, f"generate {process_string} table"):
        logger.info('data template: ' + data_config_file)
        logger.info('config_file: ' + app_config_file)

        # load configs
        cfg = ConfigSet(name='DATA_CFG', config_file=data_config_file)
        cfg = ConfigSet(name='APP_CFG',  config_file=app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        # shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        # shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        exit_code = create_geojson_table()
        if exit_code != 0:
            logger.error("GEOJSON table creation had errors. Exiting.")
            return


def create_geojson_table():
    client = Client(n_workers=20, threads_per_worker=1, memory_limit=0.1)
    print (client)

    """
    Vectorizes npy array annotation file into polygons and builds GeoJson with the polygon features.
    Creates a geojson file per labelset.
    """

    # get application and data config variables
    cfg = ConfigSet()
    SLIDEVIEWER_API_URL     = cfg.get_value('DATA_CFG::SLIDEVIEWER_API_URL')
    SLIDEVIEWER_CSV_FILE    = cfg.get_value('DATA_CFG::SLIDEVIEWER_CSV_FILE')
    PROJECT_ID              = cfg.get_value('DATA_CFG::PROJECT_ID')
    LANDING_PATH            = cfg.get_value('DATA_CFG::LANDING_PATH')
    TMP_ZIP_DIR_NAME        = cfg.get_value('DATA_CFG::REQUESTOR_DEPARTMENT') + '_tmp_zips'
    TMP_ZIP_DIR             = os.path.join(LANDING_PATH, TMP_ZIP_DIR_NAME)
    SLIDE_BMP_DIR           = os.path.join(LANDING_PATH, 'regional_bmps')
    SLIDE_NPY_DIR           = os.path.join(LANDING_PATH, 'regional_npys')
    TABLE_OUT_DIR           = os.path.join(LANDING_PATH, 'tables', 'REGIONAL_METADATA_RESULTS')

    os.makedirs(TABLE_OUT_DIR, exist_ok=True)
    print ("Table output directory =", TABLE_OUT_DIR)

    # setup variables needed for build geojson UDF
    contour_level       = cfg.get_value('DATA_CFG::CONTOUR_LEVEL')
    polygon_tolerance   = cfg.get_value('DATA_CFG::POLYGON_TOLERANCE')
    
    # fetch full set of slideviewer slides for project
    slides = fetch_slide_ids(SLIDEVIEWER_API_URL, PROJECT_ID, const.CONFIG_LOCATION(cfg), SLIDEVIEWER_CSV_FILE)
    df = pd.DataFrame(data=np.array(slides),columns=["slideviewer_path", "slide_id", "sv_project_id"])

    # get users and labelsets for df explosion
    all_users_list = cfg.get_value('DATA_CFG::USERS')
    all_labelsets  = cfg.get_value('DATA_CFG::LABEL_SETS')

    bmp_jobs = []
    for _, row in df.iterrows():
        bmp_future = client.submit (check_slideviewer_and_download_bmp, row.sv_project_id, row.slideviewer_path, row.slide_id, all_users_list, SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR)
        bmp_jobs.append( bmp_future )

    json_jobs = []
    for bmp_future in as_completed(bmp_jobs):
        if bmp_future.result() is not None:
            json_future = client.submit (convert_slide_bitmap_to_geojson, bmp_future, all_labelsets, SLIDE_NPY_DIR, contour_level, polygon_tolerance)
            json_jobs.append ( json_future )

    for json_future in as_completed(json_jobs):
        if json_future.result() is not None:
            slide_id, data = json_future.result()
            pd.DataFrame(data).to_csv(f"{TABLE_OUT_DIR}/regional_annot_slice_slide={slide_id}.csv")

    client.shutdown()
    del client

    return 0

if __name__ == "__main__":
    cli()
