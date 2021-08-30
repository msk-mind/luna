"""
Data Transfer / Unpack binaries
- load feature table
- get binaries out into png
- rename pngs
- column_name/accession#/instance#.png
"""
import os, time
import shutil

import click
import pandas as pd
from PIL import Image 
from io import BytesIO
from pyspark.sql.functions import countDistinct

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.common.custom_logger import init_logger
import luna.common.constants as const


logger = init_logger()
logger.info("Starting luna.radiology.unpack_images.unpack")


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
        This module generates a delta table with embedded png binaries based on the input and output parameters
         specified in the data_config_file.

        Example:
            python3 -m luna.radiology.feature_table.unpack \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file>
    """
    start_time = time.time()

    cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

    # copy app and data configuration to destination config dir
    config_location = cfg.get_value(path=const.DATA_CFG+"::DESTINATION_PATH")
    os.makedirs(config_location, exist_ok=True)

    shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
    shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
    logger.info("config files copied to %s", config_location)

    binary_to_png(cfg)

    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))

def binary_to_png(cfg):
    """
    Load given table, unpack dicom, overlay images and save t.
    """
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='unpack')
    spark.conf.set('spark.sql.execution.arrow.pyspark.enabled','false')

    TABLE_PATH = cfg.get_value(path=const.DATA_CFG+"::TABLE_PATH")
    DESTINATION_PATH = cfg.get_value(path=const.DATA_CFG+"::DESTINATION_PATH")
    COLUMN_NAME = cfg.get_value(path=const.DATA_CFG+"::COLUMN_NAME")
    IMAGE_WIDTH = int(cfg.get_value(path=const.DATA_CFG+"::IMAGE_WIDTH"))
    IMAGE_HEIGHT = int(cfg.get_value(path=const.DATA_CFG+"::IMAGE_HEIGHT"))

    df = spark.read.format("delta").load(TABLE_PATH)

    # find edge cases with more than 1 annotations
    # (sometimes both L/R organs have tumor, and we end up with 2 annotations per accesion.)
    multiple_annotations = df.groupby("metadata.AccessionNumber") \
        .agg(countDistinct("scan_annotation_record_uuid").alias("count")) \
        .filter("count > 1").toPandas()

    multiple_cases = multiple_annotations['AccessionNumber'].to_list()

    # unpack COLUMN_NAME
    for index, row in df.toPandas().iterrows():
        # mode set to L for b/w images, RGB for colored images.
        if "dicom" == COLUMN_NAME.lower():
            mode = "L"
        if "overlay" == COLUMN_NAME.lower():
            mode = "RGB"

        image = Image.frombytes(mode, (IMAGE_WIDTH, IMAGE_HEIGHT), bytes(row[COLUMN_NAME]))

        image_dir = os.path.join(DESTINATION_PATH, COLUMN_NAME, row["metadata"]["AccessionNumber"])

        if row["metadata"]["AccessionNumber"] in multiple_cases and row.label:
            image_dir = os.path.join(DESTINATION_PATH, COLUMN_NAME, row["metadata"]["AccessionNumber"] + "_" + row.label)

        os.makedirs(image_dir, exist_ok=True)

        # save image to png
        image.save(os.path.join(image_dir, str(index)+".png"))

if __name__ == "__main__":
    cli()

