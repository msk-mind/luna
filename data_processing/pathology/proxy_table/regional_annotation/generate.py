'''
Created on January 30, 2021

@author: pashaa@mskcc.org
'''
import glob
import pathlib
import shutil
import sys
import zipfile

import click
import requests
from filehash import FileHash

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
import numpy as np

import yaml, os

from data_processing.pathology.proxy_table.regional_annotation.slideviewer_client import fetch_slide_ids

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

    :param bmp_file - /path/to/image.bmp
    :param output_folder - /path/to/output/folder


    Troubleshooting:
        Make sure Pillow is upgraded to version 8.0.0 if getting an Unsupported BMP Size OS Error
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



def download_zip(url, zipfile_path, chunk_size=128):
    '''
    # useful: https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url

    Downloads zip from the specified URL and saves it to the specified file path.

    :url - url to download zip from
    :zipfile_path - file path where zipfile should be saved
    :return True if zipfile downloaded and saved, else false
    '''
    response = requests.get(url, stream=True)
    with open(zipfile_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk == b'Label image not found.':  # message from slideviewer
                return False
            else:
                fd.write(chunk)
        return True

    return False

def unzip(zipfile_path):
    '''

    :param zipfile_path: path of zipfile to unzip
    :return: readfile pointer to unzippped file if successfully unzippped, else None
    '''
    print(" +- UNZIP " + zipfile_path)
    try:
        return zipfile.ZipFile(zipfile_path)  # returns read file pointer
    except zipfile.BadZipFile as err:
        print(err)
        print(' +- ERROR Dumping invalid Zipfile ' + zipfile_path + ':')
        with open(zipfile_path, 'r') as zipf:
            print(zipf.read())

        return None


def process_regional_annotation_slide_row_pandas(row: pd.DataFrame) -> pd.DataFrame:
    '''
    Downloads regional annotation bmps for each row in dataframe and saves the bmp to disc.

    :return updated dataframe with bmp metadata
    '''

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

    #url = "https://slides-res.mskcc.org/slides/" + str(user) + "@mskcc.org/projects;" + str(
    #    PROJECT_ID) + ';' + query + "/getLabelFileBMP"


    print("Pulling   ", url)
    print(" +- TO    ", bmp_dest_path)

    success = download_zip(url, zipfile_path)

    row["bmp_record_uuid"] = np.nan
    row["bmp_filepath"] = np.nan

    if not success:
        os.remove(zipfile_path)
        print(" +- Label annotation file does not exist for slide and user.")
        return row

    unzipped_file_descriptor = unzip(zipfile_path)

    if unzipped_file_descriptor is None:
        return row


    # create bmp file from unzipped file
    os.makedirs(os.path.dirname(bmp_dest_path), exist_ok=True)
    with open(bmp_dest_path, "wb") as f:
        f.write(unzipped_file_descriptor.read("labels.bmp"))

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

def create_proxy_table(app_config_file, data_config_file):
    '''
    Creates the pathology annotations proxy table with information contained in the specified data_config_file

    :param data_config_file: data configuration
    :param app_config_file: app configuration
    :return: exit_code = 0 if successful, > 0 if unsuccessful
    '''
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    exit_code = 0
    cfg = ConfigSet()
    spark = SparkConfig().spark_session(
                                 config_name=const.APP_CFG,
                                 app_name="data_processing.pathology.proxy_table.annotation.generate")

    slides = fetch_slide_ids()
    df = pd.DataFrame(data=np.array(slides), columns=["slideviewer_path", "slide_id"])

    # initialize columns
    df["sv_project_id"] = cfg.get_value(const.DATA_CFG+'::PROJECT_ID')
    df["bmp_filepath"] = ''
    df["user"] = [cfg.get_value(const.DATA_CFG+'::USERS')] * len(df)
    current_time_utc = pd.to_datetime('now')
    current_time = pd.Timestamp(current_time_utc, tz='UTC').tz_convert('US/Eastern')
    df["date_added"] = current_time
    df["date_updated"] = current_time
    df["bmp_record_uuid"] = ''
    df["latest"] = True

    df = df.explode('user')

    df["bmp_filepath"] = ""
    df["bmp_record_uuid"] = ""
    # apply in pandas columns
    SLIDE_BMP_DIR = os.path.join(cfg.get_value(const.DATA_CFG+'::LANDING_PATH'), "regional_bmps")

    df["SLIDE_BMP_DIR"] = SLIDE_BMP_DIR
    TMP_ZIP_DIR = cfg.get_value(const.DATA_CFG+'::REQUESTOR_DEPARTMENT') + "_tmp_zips"
    df["TMP_ZIP_DIR"] = os.path.join(cfg.get_value(const.DATA_CFG+'::LANDING_PATH'), TMP_ZIP_DIR)
    df["SLIDEVIEWER_API_URL"] = cfg.get_value(const.DATA_CFG + '::SLIDEVIEWER_API_URL')

    def cleanup_tmp_zip_dir():
        if os.path.exists(df["TMP_ZIP_DIR"]):
            shutil.rmtree(df["TMP_ZIP_DIR"])
            logger.info("deleted tmp zip dir: "+ df["TMP_ZIP_DIR"])

    df = spark.createDataFrame(df)

    df = df.groupby(["slideviewer_path", "user"]) \
        .applyInPandas(process_regional_annotation_slide_row_pandas, schema=df.schema)
    df = df.toPandas()
    df = df.drop(columns=['SLIDE_BMP_DIR', 'TMP_ZIP_DIR', 'SLIDEVIEWER_API_URL'])

    # get slides with non-empty annotations
    df = df.replace("", np.nan)
    df = df.dropna()

    # add to graph
    # df.apply(lambda x: add_bmp_triple(x.slide_id, x.bmp_record_uuid), axis=1)

    # convert annotation bitmaps to numpy arrays
    SLIDE_NPY_DIR = os.path.join(cfg.get_value(const.DATA_CFG+'::LANDING_PATH'), 'regional_npys')
    os.makedirs(SLIDE_NPY_DIR, exist_ok=True)

    # convert to numpy
    df["npy_filepath"] = df.apply(lambda x: convert_bmp_to_npy(x.bmp_filepath, SLIDE_NPY_DIR), axis=1)

    spark_bitmask_df = spark.createDataFrame(df)

    print("Table after adding bitmap to npy conversion:")
    print(spark_bitmask_df.show(truncate=False))

    # create proxy bitmask table
    # update main table if exists, otherwise create main table
    BITMASK_TABLE_PATH = os.path.join(cfg.get_value(const.DATA_CFG+'::LANDING_PATH') , 'table/regional_bitmask')
    write_uri = os.environ["HDFS_URI"]
    save_path = os.path.join(write_uri, BITMASK_TABLE_PATH)
    if not os.path.exists(BITMASK_TABLE_PATH):
        logger.info("creating new bitmask table")
        os.makedirs(BITMASK_TABLE_PATH)
        spark_bitmask_df.coalesce(48).write.format("delta").save(save_path)
    else:
        logger.info("updating existing bitmask table")
        from delta.tables import DeltaTable
        bitmask_table = DeltaTable.forPath(spark, save_path)
        bitmask_table.alias("main_bitmask_table") \
            .merge(spark_bitmask_df.alias("bmp_annotation_updates"),
                   "main_bitmask_table.bmp_record_uuid = bmp_annotation_updates.bmp_record_uuid") \
            .whenMatchedUpdate(set={"date_updated": "bmp_annotation_updates.date_updated"}) \
            .whenNotMatchedInsertAll() \
            .execute()

    return exit_code


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

        # write template file to manifest_yaml under LANDING_PATH
        # todo: write to hdfs without using local gpfs/
        hdfs_path = os.environ['MIND_GPFS_DIR']
        landing_path = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')

        full_landing_path = os.path.join(hdfs_path, landing_path)
        if not os.path.exists(full_landing_path):
            os.makedirs(full_landing_path)
        shutil.copy(data_config_file, os.path.join(full_landing_path, "manifest.yaml"))
        logger.info("template file copied to" + os.path.join(full_landing_path, "manifest.yaml"))

        # create proxy table
        exit_code = create_proxy_table(app_config_file, data_config_file)
        if exit_code != 0:
            logger.error("Delta table creation had errors. Exiting.")
            return


if __name__ == "__main__":
    cli()