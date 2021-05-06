import shutil
import click
import yaml, os, json
from datetime import datetime

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
# from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.common.utils import get_absolute_path
from data_processing.pathology.common.utils import get_labelset_keys
import pandas as pd
import numpy as np

from data_processing.common.utils import get_absolute_path
from data_processing.pathology.common.slideviewer_client import fetch_slide_ids, download_zip, unzip
from data_processing.pathology.common.build_geojson import build_default_geojson_from_annotation, build_all_geojsons_from_default, concatenate_regional_geojsons
import dask
from dask.distributed import Client, as_completed

from PIL import Image

import copy, shapely
from io import BytesIO
from skimage import measure

from filehash import FileHash



def get_slide_bitmap(full_filename, user, slide_id, SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR, sv_project_id):

    print(f" >>>>>>> Processing [{full_filename}] <<<<<<<<")

    full_filename_without_ext = full_filename.replace(".svs", "")

    bmp_dirname = os.path.join(SLIDE_BMP_DIR, full_filename_without_ext.replace(";", "_"))
    bmp_dest_path = os.path.join(bmp_dirname, str(slide_id) + '_' + user + '_annot.bmp')

    if os.path.exists(bmp_dest_path):
        print("Removing temporary file "+bmp_dest_path)
        os.remove(bmp_dest_path)

    # download bitmap file using api (from brush and fill tool), download zips into TMP_ZIP_DIR
    os.makedirs(TMP_ZIP_DIR, exist_ok=True)
    zipfile_path = os.path.join(TMP_ZIP_DIR, full_filename_without_ext + "_" + user + ".zip")

    url = SLIDEVIEWER_API_URL +'slides/'+ str(user) + '@mskcc.org/projects;' + str(sv_project_id) + ';' + full_filename + '/getLabelFileBMP'

    # print("Pulling   ", url)
    # print(" +- TO    ", bmp_dest_path)

    success = download_zip(url, zipfile_path)

    bmp_record_uuid = 'n/a'
    bmp_filepath = 'n/a'

    if not success:
        os.remove(zipfile_path)
        # print(" +- Label annotation file does not exist for slide and user.")
        return (bmp_record_uuid, bmp_filepath)

    unzipped_file_descriptor = unzip(zipfile_path)

    if unzipped_file_descriptor is None:
        return (bmp_record_uuid, bmp_filepath)


    # create bmp file from unzipped file
    os.makedirs(os.path.dirname(bmp_dest_path), exist_ok=True)
    with open(bmp_dest_path, "wb") as ff:
        ff.write(unzipped_file_descriptor.read("labels.bmp"))  # all bmps from slideviewer are called labels.bmp

    print(" +- Added slide " + str(slide_id) + " to " + str(bmp_dest_path) + "  * * * * ")

    bmp_hash = FileHash('sha256').hash_file(bmp_dest_path)
    bmp_record_uuid = f'SVBMP-{bmp_hash}'
    bmp_filepath = bmp_dirname + '/' + slide_id + '_' + user + '_' + "thisis" + '_annot.bmp'
    os.rename(bmp_dest_path, bmp_filepath)
    # print(" +- Generated record " + bmp_filepath)

    # cleanup
    if os.path.exists(zipfile_path):
        os.remove(zipfile_path)

    return (bmp_record_uuid, bmp_filepath)

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

    Image.MAX_IMAGE_PIXELS = 5000000000

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


# from pyspark.sql.functions import udf, lit, col, first, last, desc, array, to_json, collect_list, current_timestamp, explode
# from pyspark.sql.window import Window
# from pyspark.sql.types import StringType, IntegerType, ArrayType, MapType, StructType, StructField


os.environ['OPENBLAS_NUM_THREADS'] = '1'
logger = init_logger()


# Base template for geoJSON file
geojson_base = {
    "type": "FeatureCollection",
    "features": []
}

# max amount of time for a geojson to be generated. if generation surpasses this limit, it is likely the annotation file is
# too large or they may be annotation artifacts present in the slide. currently set at 30 minute timeout
TIMEOUT_SECONDS = 1800


# @click.command()
# @click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
#               help="path to yaml file containing data input and output parameters. "
#                    "See ./data_config.yaml.template")
# @click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
#               help="path to yaml file containing application runtime parameters. "
#                    "See ./app_config.yaml.template")
# @click.option('-p', '--process_string', default='geojson',
#               help='process to run or replay: e.g. geojson OR concat')
# def cli(data_config_file, app_config_file, process_string):
#     """
#         This module generates a delta table with geojson pathology data based on the input and output parameters
#          specified in the data_config_file.

#         Example:
#             python3 -m data_processing.pathology.refined_table.regional_annotation.generate \
#                      --data_config_file <path to data config file> \
#                      --app_config_file <path to app config file> \
#                      --process_string geojson
#     """
#     with CodeTimer(logger, f"generate {process_string} table"):
#         logger.info('data template: ' + data_config_file)
#         logger.info('config_file: ' + app_config_file)

#         # load configs


#         # copy app and data configuration to destination config dir
#         config_location = const.CONFIG_LOCATION(cfg)
#         os.makedirs(config_location, exist_ok=True)

#         # shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
#         # shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
#         # logger.info("config files copied to %s", config_location)

#         if 'geojson' == process_string:
#             exit_code = create_geojson_table()
#             if exit_code != 0:
#                 logger.error("GEOJSON table creation had errors. Exiting.")
#                 return

#         # if 'concat' == process_string:
#         #     exit_code = create_concat_geojson_table()
#         #     if exit_code != 0:
#         #         logger.error("CONCAT-GEOJSON table creation had errors. Exiting.")
#         #         return


def create_geojson_table():
    cfg = ConfigSet(name=const.DATA_CFG, config_file='test_dask_config.yaml')
    cfg = ConfigSet(name=const.APP_CFG,  config_file='../configs/config.yaml')

    """
    Vectorizes npy array annotation file into polygons and builds GeoJson with the polygon features.
    Creates a geojson file per labelset.
    """
    

    # get application and data config variables
    cfg = ConfigSet()
    SLIDEVIEWER_API_URL = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_API_URL')
    SLIDEVIEWER_CSV_FILE = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_CSV_FILE')
    PROJECT_ID = cfg.get_value(path=const.DATA_CFG + '::PROJECT_ID')
    LANDING_PATH = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')
    SLIDE_BMP_DIR = os.path.join(LANDING_PATH, 'regional_bmps')
    TMP_ZIP_DIR_NAME = cfg.get_value(const.DATA_CFG + '::REQUESTOR_DEPARTMENT') + '_tmp_zips'
    TMP_ZIP_DIR = os.path.join(LANDING_PATH, TMP_ZIP_DIR_NAME)
    SLIDE_NPY_DIR = os.path.join(LANDING_PATH, 'regional_npys')

    # setup variables needed for build geojson UDF
    contour_level = cfg.get_value(path=const.DATA_CFG+'::CONTOUR_LEVEL')
    polygon_tolerance = cfg.get_value(path=const.DATA_CFG+'::POLYGON_TOLERANCE')
    

    # fetch full set of slideviewer slides for project
    # slides = fetch_slide_ids(SLIDEVIEWER_API_URL, PROJECT_ID, const.CONFIG_LOCATION(cfg), SLIDEVIEWER_CSV_FILE)
    # df = pd.DataFrame(data=np.array(slides),columns=["slideviewer_path", "slide_id", "sv_project_id"])

    # FOR TESTTING, REMOVE
    df = pd.read_csv('/home/pateld6/data-processing/test_slides.csv')
    df = df.head(2)

    # get users and labelsets for df explosion
    users = cfg.get_value(const.DATA_CFG + '::USERS')
    all_labelsets = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')
    
    df['users'] = [users] * len(df)
    # df = df.explode('user')

    df['all_labelsets'] = [all_labelsets] * len(df)
    # df = df.explode('labelset')

    # reindexing
    df = df.reset_index(drop=True)

    # adding universal columns
    now = datetime.now() 
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    df['date_updated'] = date_time
    df['SLIDE_BMP_DIR'] = SLIDE_BMP_DIR
    df['TMP_ZIP_DIR'] = TMP_ZIP_DIR
    df['SLIDEVIEWER_API_URL'] = SLIDEVIEWER_API_URL
    df['contour_level'] = contour_level
    df['polygon_tolerance'] = polygon_tolerance
    df['SLIDE_NPY_DIR'] = SLIDE_NPY_DIR 




    # define client
    # 
    # remove

    # projects;134;2018;HobS18-062598116001;1216705.svs/getLabelFileBMP
    # df = df[df['slide_id'] == '2550967']
    # process_slide()

    client = Client("tcp://10.254.130.16:8786")
    futures = client.map(process_slide, df['slideviewer_path'], df['slide_id'], df['users'], df['all_labelsets'], df['SLIDE_NPY_DIR'], df['SLIDE_BMP_DIR'], df['SLIDEVIEWER_API_URL'], df['TMP_ZIP_DIR'], df['sv_project_id'], df['contour_level'], df['polygon_tolerance'])

    results = client.gather(futures)
    df[["bmp_record_uuid","bmp_filepath","npy_filepath","geojson"]] = list(results)
    print(df)
    # df["geojson_uuid"]

    


def process_slide(slideviewer_path, slide_id, users, all_labelsets, SLIDE_NPY_DIR, SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR, sv_project_id, contour_level, polygon_tolerance):
    slide_id = str(slide_id)    

    outputs = []
    output_dict_base = {"slide_id": slide_id, "user": "n/a", "bmp_filepath": 'n/a', "npy_filepath": 'n/a', "geojson": 'n/a'}
    geojson_table_outs = []
    concat_geojson_table_outs = []

    for user in users:
        # download bitmap
        bmp_record_uuid, bmp_filepath = get_slide_bitmap(slideviewer_path, user, slide_id, SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR, sv_project_id)
        # convert to npy
        if bmp_record_uuid != 'n/a' or bmp_filepath != 'n/a':

            output_dict = copy.deepcopy(output_dict_base)
            output_dict['user'] = user
            output_dict["bmp_filepath"] = bmp_filepath
            outputs.append(output_dict)
            
    # at this point if outputs is empty, return early
    if len(outputs) == 0:
        return None

    for user_annotation in outputs:
        bmp_filepath = user_annotation['bmp_filepath']
        npy_filepath = convert_bmp_to_npy(bmp_filepath, SLIDE_NPY_DIR)
        user_annotation['npy_filepath'] = npy_filepath

    # build geojsons
    for user_annotation in outputs:
        npy_filepath = user_annotation['npy_filepath']
        default_annotation_geojson = build_default_geojson_from_annotation(npy_filepath, all_labelsets, contour_level, polygon_tolerance)

        if not default_annotation_geojson:
            print("Error when building geojson")
            return None 


        user_annotation['geojson'] = default_annotation_geojson

    
    for user_annotation in outputs:
        default_annotation_geojson = user_annotation['geojson']
        labelset_name_to_labelset_specific_geojson = build_all_geojsons_from_default(default_annotation_geojson, all_labelsets, contour_level, polygon_tolerance)
        for labelset_name, geojson in labelset_name_to_labelset_specific_geojson.items():
            geojson_table_out_entry = copy.deepcopy(user_annotation)
            geojson_table_out_entry['labelset'] = labelset_name
            geojson_table_out_entry['geojson'] = geojson
            geojson_table_outs.append(geojson_table_out_entry)
    

    
    geojsons_to_concat = [json.dumps(user_annotation['geojson']) for user_annotation in outputs]
    concat_default_annotation_geojson = concatenate_regional_geojsons(geojsons_to_concat)
    labelset_name_to_labelset_specific_geojson = build_all_geojsons_from_default(concat_default_annotation_geojson, all_labelsets, contour_level, polygon_tolerance)
    for labelset_name, geojson in labelset_name_to_labelset_specific_geojson.items():
        concat_geojson_table_out_entry = copy.deepcopy(output_dict_base)
        concat_geojson_table_out_entry['user'] = "CONCAT"
        concat_geojson_table_out_entry['labelset'] = labelset_name
        concat_geojson_table_out_entry['geojson'] = geojson
        concat_geojson_table_outs.append(concat_geojson_table_out_entry)
    
    return slide_id, geojson_table_outs + concat_geojson_table_outs


    # build concat geojson

    
    # return (bmp_record_uuid,bmp_filepath,npy_filepath,geojson)
    # print( " >>>>>>> Finished annotation processing, printing resulting geojson...", geojson[:200])
    
    #     return (bmp_filepath, npy_filepath, geojson)




    # populate columns, bmp_filepath, npy_filepath, geojson
    # df = df.withColumn('bmp_filepath', lit('')) \
    #     .withColumn('users', array([lit(user) for user in cfg.get_value(const.DATA_CFG + '::USERS')])) \
    #     .withColumn('date_updated', current_timestamp()) \
    #     .withColumn('geojson_record_uuid', lit('')) \
    #     .withColumn('SLIDE_BMP_DIR', lit(os.path.join(LANDING_PATH, 'regional_bmps'))) \
    #     .withColumn('TMP_ZIP_DIR', lit(os.path.join(LANDING_PATH, TMP_ZIP_DIR))) \
    #     .withColumn('SLIDEVIEWER_API_URL', lit(cfg.get_value(const.DATA_CFG + '::SLIDEVIEWER_API_URL'))) \


    # old


    exit_code = 0

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG,
                                        app_name="data_processing.pathology.refined_table.annotation.generate")
    # disable broadcast join to avoid timeout
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

    # load paths from configs
    bitmask_table_path = const.TABLE_LOCATION(cfg, is_source=True)
    geojson_table_path = const.TABLE_LOCATION(cfg)

    df = spark.read.format("delta").load(bitmask_table_path)

    # explode table by labelsets
    labelsets = get_labelset_keys()
    labelset_column = array([lit(key) for key in labelsets])

    df = df.withColumn("labelset_list", labelset_column)
    # explode labelsets
    df = df.select("slideviewer_path", "slide_id", "sv_project_id", "bmp_record_uuid", "user", "npy_filepath", explode("labelset_list").alias("labelset"))

    # setup variables needed for build geojson UDF
    contour_level = cfg.get_value(path=const.DATA_CFG+'::CONTOUR_LEVEL')
    polygon_tolerance = cfg.get_value(path=const.DATA_CFG+'::POLYGON_TOLERANCE')

    # populate geojson and geojson_record_uuid
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/EnsureByteContext.py"))
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/utils.py"))
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../common/build_geojson.py"))
    
    label_config = cfg.get_value(path=const.DATA_CFG+'::LABEL_SETS')

    df = df.withColumn("label_config", lit(str(label_config))) \
            .withColumn("contour_level", lit(contour_level)) \
            .withColumn("polygon_tolerance", lit(polygon_tolerance)) \
            .withColumn("geojson", lit(""))

    print(df.select("label_config").collect())

    df = df.groupby(["bmp_record_uuid", "labelset"]).applyInPandas(build_geojson_from_annotation, schema = df.schema)

    # drop empty geojsons that may have been created
    df = df.filter("geojson != ''")

    # populate uuid

    geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())
    spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/EnsureByteContext.py"))
    df = df.withColumn("geojson_record_uuid", geojson_record_uuid_udf("geojson", array(lit("SVGEOJSON"), "labelset")))

    # build refined table by selecting columns from output table
    geojson_df = df.select("sv_project_id", "slideviewer_path", "slide_id", "bmp_record_uuid", "user", "labelset", "geojson", "geojson_record_uuid")
    geojson_df = geojson_df.withColumn("latest", lit(True))        \
                         .withColumn("date_added", current_timestamp())    \
                         .withColumn("date_updated", current_timestamp())
    # update geojson delta table
    geojson_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(geojson_table_path)

    logger.info("Finished building Geojson table.")

    return exit_code


# def create_concat_geojson_table():
#     """
#     Aggregate geojson features for each labelset, in case there are annotations from multiple users.
#     """
#     exit_code = 0

#     cfg = ConfigSet()
#     spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.refined_table.annotation.generate")
#     spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

#     # load paths from config
#     geojson_table_path = const.TABLE_LOCATION(cfg, is_source=True)
#     concat_geojson_table_path = const.TABLE_LOCATION(cfg)

#     concatgeojson_df = spark.read.format("delta").load(geojson_table_path)
#     # only use latest annotations for concatenating.
#     concatgeojson_df = concatgeojson_df.filter("latest")

#     # make geojson string list for slide + labelset
#     concatgeojson_df = concatgeojson_df \
#         .select("sv_project_id", "slideviewer_path", "slide_id", "labelset", "geojson") \
#         .groupby(["sv_project_id", "slideviewer_path", "slide_id", "labelset"]) \
#         .agg(collect_list("geojson").alias("geojson_list"))

#     # set up udfs
#     spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/EnsureByteContext.py"))
#     spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../../common/utils.py"))
#     spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../common/build_geojson.py"))
#     from utils import generate_uuid_dict
#     from build_geojson import concatenate_regional_geojsons
#     concatenate_regional_geojsons_udf = udf(concatenate_regional_geojsons, geojson_struct)
#     concat_geojson_record_uuid_udf = udf(generate_uuid_dict, StringType())

#     # cache to not have udf called multiple times
#     concatgeojson_df = concatgeojson_df.withColumn("concat_geojson", concatenate_regional_geojsons_udf("geojson_list")).cache()

#     concatgeojson_df = concatgeojson_df \
#         .drop("geojson_list") \
#         .withColumn("concat_geojson_record_uuid", concat_geojson_record_uuid_udf(to_json("concat_geojson"), array(lit("SVCONCATGEOJSON"), "labelset")))
    
#     concatgeojson_df = concatgeojson_df.withColumn("latest", lit(True))   \
#                             .withColumn("date_added", current_timestamp())    \
#                             .withColumn("date_updated", current_timestamp())

#     # update concatenation geojson delta table
#     from delta.tables import DeltaTable
#     concatgeojson_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(concat_geojson_table_path)

#     logger.info("Finished building Concatenation table.")

#     return exit_code


if __name__ == "__main__":
    
    # future = client.submit(print, "Test")
    # future.result()
    # cli()
    create_geojson_table()
