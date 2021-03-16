'''
Created on January 30, 2021

@author: pashaa@mskcc.org
'''
import pathlib
import shutil

import click
from filehash import FileHash
from pyspark.sql.functions import array, current_timestamp, explode, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

import pandas as pd
import numpy as np

import os

from data_processing.pathology.common.slideviewer_client import fetch_slide_ids

logger = init_logger()

DATA_SCHEMA_FILE = os.path.join(
                      pathlib.Path(__file__).resolve().parent,
                      'data_config_schema.yml')

from PIL import Image
Image.MAX_IMAGE_PIXELS = 5000000000


def convert_bmp_to_npy(bmp_file, output_folder):
    """
    Reads a bmp file and creates friendly numpy ndarray file in the uint8 format in the output
    directory specified, with extention .annot.npy

    Troubleshooting:
        Make sure Pillow is upgraded to version 8.0.0 if getting an Unsupported BMP Size OS Error

    :param bmp_file - /path/to/image.bmp
    :param output_folder - /path/to/output/folder
    :return filepath to file containing numpy array
    """
    if not '.bmp' in bmp_file:
        return ''

    new_image_name = os.path.basename(bmp_file).replace(".bmp", ".npy")
    bmp_caseid_folder = os.path.basename(os.path.dirname(bmp_file))
    output_caseid_folder = os.path.join(output_folder, bmp_caseid_folder)

    if not os.path.exists(output_caseid_folder):
        os.makedirs(output_caseid_folder)

    output_filepath = os.path.join(output_caseid_folder, new_image_name)

    np.save(output_filepath, np.array(Image.open(bmp_file)))
    return output_filepath


def process_regional_annotation_slide_row_pandas(row: pd.DataFrame) -> pd.DataFrame:
    '''
    Downloads regional annotation bmps for each row in dataframe and saves the bmp to disc.

    :return updated dataframe with bmp metadata
    '''
    from slideviewer_client import download_zip, unzip

    full_filename = row.slideviewer_path.item()
    user = row.user.item()

    print(f" >>>>>>> Processing [{full_filename}] <<<<<<<<")

    full_filename_without_ext = full_filename.replace(".svs", "")

    slide_id = row.slide_id.item()

    bmp_dirname = os.path.join(row.SLIDE_BMP_DIR.item(), full_filename_without_ext.replace(";", "_"))
    bmp_dest_path = os.path.join(bmp_dirname, str(slide_id) + '_' + user + '_annot.bmp')

    if os.path.exists(bmp_dest_path):
        print("Removing temporary file "+bmp_dest_path)
        os.remove(bmp_dest_path)

    # download bitmap file using api (from brush and fill tool), download zips into TMP_ZIP_DIR
    TMP_ZIP_DIR = row.TMP_ZIP_DIR.item()
    os.makedirs(TMP_ZIP_DIR, exist_ok=True)
    zipfile_path = os.path.join(TMP_ZIP_DIR, full_filename_without_ext + "_" + user + ".zip")

    url = row.SLIDEVIEWER_API_URL.item() +'slides/'+ str(user) + '@mskcc.org/projects;' + str(row.sv_project_id.item()) + ';' + full_filename + '/getLabelFileBMP'

    print("Pulling   ", url)
    print(" +- TO    ", bmp_dest_path)

    success = download_zip(url, zipfile_path)

    row["bmp_record_uuid"] = 'n/a'
    row["bmp_filepath"] = 'n/a'

    if not success:
        os.remove(zipfile_path)
        print(" +- Label annotation file does not exist for slide and user.")
        return row

    unzipped_file_descriptor = unzip(zipfile_path)

    if unzipped_file_descriptor is None:
        return row


    # create bmp file from unzipped file
    os.makedirs(os.path.dirname(bmp_dest_path), exist_ok=True)
    with open(bmp_dest_path, "wb") as ff:
        ff.write(unzipped_file_descriptor.read("labels.bmp"))  # all bmps from slideviewer are called labels.bmp

    print(" +- Added slide " + str(slide_id) + " to " + str(bmp_dest_path) + "  * * * * ")

    bmp_hash = FileHash('sha256').hash_file(bmp_dest_path)
    row["bmp_record_uuid"] = f'SVBMP-{bmp_hash}'
    row["bmp_filepath"] = bmp_dirname + '/' + slide_id + '_' + user + '_' + row["bmp_record_uuid"].item() + '_annot.bmp'
    os.rename(bmp_dest_path, row["bmp_filepath"].item())
    print(" +- Generated record " + row["bmp_record_uuid"].item())

    # cleanup
    if os.path.exists(zipfile_path):
        os.remove(zipfile_path)

    return row

def create_proxy_table():
    '''
    Creates the pathology annotations proxy table with information contained in the specified data_config_file

    :param data_config_file: data configuration
    :param app_config_file: app configuration
    :return: exit_code = 0 if successful, > 0 if unsuccessful
    '''
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # pd.set_option('display.max_colwidth', -1)

    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(
                                 config_name=const.APP_CFG,
                                 app_name="data_processing.pathology.proxy_table.annotation.generate")

    SLIDEVIEWER_API_URL = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_API_URL')
    SLIDEVIEWER_CSV_FILE = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_CSV_FILE')
    PROJECT_ID = cfg.get_value(path=const.DATA_CFG + '::PROJECT_ID')
    LANDING_PATH = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')
    slides = fetch_slide_ids(SLIDEVIEWER_API_URL, PROJECT_ID, LANDING_PATH, SLIDEVIEWER_CSV_FILE)

    schema = StructType([
        StructField('slideviewer_path', StringType()),
        StructField('slide_id', StringType()),
        StructField('sv_project_id', IntegerType())
    ])

    df = spark.createDataFrame(slides, schema)

    # populate columns
    TMP_ZIP_DIR = cfg.get_value(const.DATA_CFG + '::REQUESTOR_DEPARTMENT') + '_tmp_zips'
    df = df.withColumn('bmp_filepath', lit('')) \
        .withColumn('users', array([lit(user) for user in cfg.get_value(const.DATA_CFG + '::USERS')])) \
        .withColumn('date_added', current_timestamp()) \
        .withColumn('date_updated', current_timestamp()) \
        .withColumn('bmp_record_uuid', lit('')) \
        .withColumn('latest', lit(True)) \
        .withColumn('SLIDE_BMP_DIR', lit(os.path.join(LANDING_PATH, 'regional_bmps'))) \
        .withColumn('TMP_ZIP_DIR', lit(os.path.join(LANDING_PATH, TMP_ZIP_DIR))) \
        .withColumn('SLIDEVIEWER_API_URL', lit(cfg.get_value(const.DATA_CFG + '::SLIDEVIEWER_API_URL'))) \

    # explore by user list
    df = df.select('slideviewer_path',
                   'slide_id',
                   'sv_project_id',
                   'bmp_filepath',
                   explode('users').alias('user'),
                   'date_added',
                   'date_updated',
                   'bmp_record_uuid',
                   'latest',
                   'SLIDE_BMP_DIR',
                   'TMP_ZIP_DIR',
                   'SLIDEVIEWER_API_URL'
                   )

    spark.sparkContext.addPyFile("./data_processing/pathology/common/slideviewer_client.py")
    df = df.groupby(['slideviewer_path', 'user']) \
        .applyInPandas(process_regional_annotation_slide_row_pandas, schema=df.schema)
    df.show()

    df = df.toPandas()
    df = df.drop(columns=['SLIDE_BMP_DIR', 'TMP_ZIP_DIR', 'SLIDEVIEWER_API_URL'])

    # get slides with non-empty annotations
    df = df.replace("n/a", np.nan)
    df = df.dropna()

    # convert annotation bitmaps to numpy arrays
    SLIDE_NPY_DIR = os.path.join(LANDING_PATH, 'regional_npys')
    os.makedirs(SLIDE_NPY_DIR, exist_ok=True)

    # convert to numpy
    df["npy_filepath"] = df.apply(lambda x: convert_bmp_to_npy(x.bmp_filepath, SLIDE_NPY_DIR), axis=1)

    spark_bitmask_df = spark.createDataFrame(df)
    spark_bitmask_df.show()

    # create proxy bitmask table
    # update main table if exists, otherwise create main table
    BITMASK_TABLE_PATH = const.TABLE_LOCATION(cfg)

    if not os.path.exists(BITMASK_TABLE_PATH):
        logger.info("creating new bitmask table")
        os.makedirs(BITMASK_TABLE_PATH)
        spark_bitmask_df.coalesce(48).write.format("delta").save(BITMASK_TABLE_PATH)
    else:
        logger.info("updating existing bitmask table")
        from delta.tables import DeltaTable
        bitmask_table = DeltaTable.forPath(spark, BITMASK_TABLE_PATH)
        bitmask_table.alias("main_bitmask_table") \
            .merge(spark_bitmask_df.alias("bmp_annotation_updates"),
                   "main_bitmask_table.bmp_record_uuid = bmp_annotation_updates.bmp_record_uuid") \
            .whenMatchedUpdate(set={"date_updated": "bmp_annotation_updates.date_updated"}) \
            .whenNotMatchedInsertAll() \
            .execute()

    # clean up TMP_ZIP_DIR
    tmp_zip_dir = os.path.join(LANDING_PATH, TMP_ZIP_DIR)
    if os.path.exists(tmp_zip_dir):
        shutil.rmtree(tmp_zip_dir)

    return exit_code


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
        This module generates a delta table with pathology data based on the input and output parameters specified in
         the data_config_file.

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
        python3 -m data_processing.pathology.proxy_table.regional_annotation.generate \
            -d {data_config_yaml} \
            -a {app_config_yaml}
    """
    with CodeTimer(logger, 'generate regional annotation proxy table'):
        # read and validate configs
        logger.info('data config: ' + data_config_file)
        logger.info('app config: ' + app_config_file)

        data_cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file, schema_file=DATA_SCHEMA_FILE)
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        # create proxy table
        exit_code = create_proxy_table()
        if exit_code != 0:
            logger.error("Delta table creation had errors. Exiting.")
            return


if __name__ == "__main__":
    cli()
