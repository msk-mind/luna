'''
Created on January 30, 2021

@author: pashaa@mskcc.org
'''
import pathlib

import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const
from data_processing.pathology.common.build_geojson_from_bitmap import build_geojson_from_bitmap_pandas, concatenate_regional_geojsons_pandas
from data_processing.pathology.common.utils import get_add_triple_str

from pyspark.sql.functions import udf, lit, col, first, last, desc
from pyspark.sql.window import Window
import pandas as pd

import yaml, os

logger = init_logger()

DATA_SCHEMA_FILE=os.path.join(
                      pathlib.Path(__file__).resolve().parent,
                      'data_config_schema.yml')


@click.command()
@click.option('-d',
              '--data_config_file',
              default=None,
              type=click.Path(exists=True),
              help="path to data config file containing information required for pathology proxy data ingestion. "
                   "See data_processing/pathology/proxy_table/annotation/data_config.yaml.template")
@click.option('-a',
              '--app_config_file',
              default='config.yaml',
              type=click.Path(exists=True),
              help="path to app config file containing application configuration. See config.yaml.template")
def cli(data_config_file, app_config_file):
    '''
    This module performs the following sequence of operations -
    1) Bitmap regional pathology tissue annotations are downloaded from SlideViewer
    2) The downloaded bitmap annotations are then converted into npy arrays
    3) A proxy table is built with the following fields.

    slideviewer_path - path to original slide image in slideviewer platform
    slide_id - synonymous with image_id
    sv_project_id - same as the project_id from the data_config.yaml,refers to the SlideViewer project number.
    bmp_filepath - file path to downloaded bmp annotation file
    annotator - id of annotator for a given annotation
    date_added - date annotation first added
    date_updated - date annotation most recently updated
    bmp_record_uuid - hash of bmp annotation file, format: SVBMP-{bmp_hash}
    npy_filepath - file path to generated npy annotation file

    Usage:
    python3 -m data_processing.pathology.proxy_table.annotation.generate \
        -d {data_config_yaml} \
        -a {app_config_yaml}
    '''
    with CodeTimer(logger, 'generate regional annotation proxy table'):
        # read and validate configs
        logger.info('data config: ' + data_config_file)
        logger.info('app config: ' + app_config_file)

        data_cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file, schema_file=DATA_SCHEMA_FILE)
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)


if __name__ == "__main__":
    cli()