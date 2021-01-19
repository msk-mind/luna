"""
This final preprocessing step
1. Loads PNG table and MHA table (2d segmentation)
2. Finds the centroid of the 2d segmentation for the scan.
3. Crop PNG images around the centroid.
4. Saves the PNG binaries in a table.
"""
import os, time, glob
import click
import numpy as np
from PIL import Image 
from medpy.io import load
from filehash import FileHash
from io import BytesIO

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, BinaryType

logger = init_logger()
logger.info("Starting data_processing.radiology.feature_table.annotation.generate")


def generate_uuid(content):

    content = BytesIO(content)

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        uuid = FileHash('sha256').hash_file(content)
 
    return "FEATURE-"+uuid

def find_centroid(path, image_w, image_h):
    """
    Find the centroid of the 2d segmentation.
    Return (x, y) center point.
    """
    # mha file path
    file_path = path.split(':')[-1]
    data, header = load(file_path)

    h, w, num_images = data.shape

    # Find the annotated slice
    xcenter, ycenter = 0, 0
    for i in range(num_images):
        seg = data[:,:,i]
        if np.any(seg):
            seg = seg.astype(float)
                    
            # find centroid using mean
            xcenter = np.argmax(np.mean(seg, axis=1))
            ycenter = np.argmax(np.mean(seg, axis=0))
            break

    # Check if h,w matches IMAGE_WIDTH, IMAGE_HEIGHT. If not, this is due to png being rescaled. So scale centers.
    image_w, image_h = int(image_w), int(image_h)
    if not h == image_h:
        xcenter = int(xcenter * image_w // w)
    if not w == image_w:
        ycenter = int(ycenter * image_h // h)

    return (int(xcenter), int(ycenter))


def crop_images(xcenter, ycenter, dicom, overlay, crop_w, crop_h, image_w, image_h):
    """
    Crop PNG images around the centroid.
    Return (dicom, overlay) binaries.
    """
    crop_w, crop_h = int(crop_w), int(crop_h)
    image_w, image_h = int(image_w), int(image_h)
    # Find xmin, ymin, xmax, ymax based on CROP_SIZE
    width_rad = crop_w // 2
    height_rad = crop_h // 2
   
    xmin, ymin, xmax, ymax = (xcenter - width_rad), (ycenter - height_rad), (xcenter + width_rad), (ycenter + height_rad)

    if xmin < 0:
        xmin = 0
        xmax = crop_w

    if xmax > image_w:
        xmin = image_w - crop_w
        xmax = image_w

    if ymin < 0:
        ymin = 0
        ymax = crop_h

    if ymax > image_h:
        ymin = image_h - crop_h
        ymax = image_h

    # Crop overlay, dicom pngs.
    dicom_img = Image.frombytes("L", (image_w, image_h), bytes(dicom)) 
    dicom_feature = dicom_img.crop((xmin, ymin, xmax, ymax)).tobytes()

    overlay_img = Image.frombytes("RGB", (image_w, image_h), bytes(overlay)) 
    overlay_feature = overlay_img.crop((xmin, ymin, xmax, ymax)).tobytes()
    
    return (dicom_feature, overlay_feature)


@click.command()
@click.option('-f', '--config_file', default='config.yaml', required=True, 
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-t', '--data_config_file', default='data_processing/radiology/feature_table/annotation/config.yaml', required=True,
    help="path to data configuration file. See data_processing/radiology/feature_table/annotation/config.yaml.template")
def cli(config_file, data_config_file):
    """
    This module generates cropped png images based on png and mha (2d segmentation) tables.
 
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.radiology.feature_table.annotation.generate \
        --data_config_file data_processing/radiology/feature_table/config.yaml \
        --config_file config.yaml
    """
    start_time = time.time()

    cfg = ConfigSet(name=const.APP_CFG, config_file=config_file)
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

    generate_feature_table(cfg)

    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_feature_table(cfg):
    """
    Load PNG and MHA table, find the centroid of the 2d segmentation, and crop PNG images around the centroid.
    """
    # setup project path
    project_path = const.PROJECT_LOCATION(cfg)

    # load dicom and seg tables
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='crop_png')

    DATASET_NAME = cfg.get_value(path=const.DATA_CFG+'::DATASET_NAME')
    CROP_WIDTH = int(cfg.get_value(path=const.DATA_CFG+'::CROP_WIDTH'))
    CROP_HEIGHT = int(cfg.get_value(path=const.DATA_CFG+'::CROP_HEIGHT'))
    IMAGE_WIDTH = int(cfg.get_value(path=const.DATA_CFG+'::IMAGE_WIDTH'))
    IMAGE_HEIGHT = int(cfg.get_value(path=const.DATA_CFG+'::IMAGE_HEIGHT'))

    png_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("PNG", DATASET_NAME))
    mha_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("MHA", DATASET_NAME))

    png_df = spark.read.format("delta").load(png_table_path) \
                  .drop("scan_annotation_record_uuid")
    
    mha_df = spark.read.format("delta").load(mha_table_path) \
                  .drop("metadata", "length", "modificationTime")

    # Find x,y centroid using MHA segmentation
    find_centroid_udf = F.udf(find_centroid, StructType([StructField("x", IntegerType()), StructField("y", IntegerType())]))
    mha_df = mha_df.withColumn("center", find_centroid_udf("path", F.lit(IMAGE_WIDTH), F.lit(IMAGE_HEIGHT))) \
                   .select(F.col("center.x").alias("x"), F.col("center.y").alias("y"), "accession_number", "scan_annotation_record_uuid", F.col("label").alias("mha_label"))
    
    logger.info("Loaded mha and png tables")

    # Join PNG and MHA tables
    columns = ["metadata", "dicom", "overlay", "png_record_uuid", "scan_annotation_record_uuid", "x","y", "label"]
    
    #TODO add L/R labels in MHD/MHA so we can match MHA/MHD, based on the accession, label
    df = mha_df.join(png_df, [png_df.metadata.AccessionNumber == mha_df.accession_number, png_df.label.eqNullSafe(mha_df.mha_label)]) \
               .select(columns) \
               .dropna(subset=["dicom", "overlay"])
    logger.info(df.count())
    logger.info("Joined mha and png tables")

    with CodeTimer(logger, 'Crop pngs and create feature table'):

        feature_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("FEATURE", DATASET_NAME))
   
        crop_images_udf = F.udf(crop_images, StructType([StructField("dicom", BinaryType()), StructField("overlay", BinaryType())]))   
        df = df.withColumn("dicom_overlay", crop_images_udf("x","y","dicom","overlay", F.lit(CROP_WIDTH), F.lit(CROP_HEIGHT), F.lit(IMAGE_WIDTH), F.lit(IMAGE_HEIGHT))) \
               .drop("dicom", "overlay") \
               .select("metadata", "png_record_uuid", "scan_annotation_record_uuid", "label",
                       F.col("dicom_overlay.dicom").alias("dicom"), F.col("dicom_overlay.overlay").alias("overlay"))
       
        logger.info("Cropped pngs")

        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        generate_uuid_udf = F.udf(generate_uuid, StringType())
        df = df.withColumn("feature_record_uuid", F.lit(generate_uuid_udf("overlay")))

        df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')) \
            .write.format("delta") \
            .mode("overwrite") \
            .save(feature_table_path)
    

if __name__ == "__main__":
    cli()

