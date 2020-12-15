### Find Dicom pipeline - which image did the tumor come from?
# 0. Annotation table / get a hold of MHAs - SeriesInstanceUID/De-ided Accession Number/SeriesNumber -- How do you get the series number 
# 1. Loop through MHA arrays and find the InstanceNumber
# 2. Query dicom table with InstanceNumber (OR image.shape[0] - InstanceNumber), De-ided Accession Number, Series Number or (Series Description = Ph1/Axial T1 FS post) and get a dicom path.
# 3. Grab dicom file
# 4. Convert and save dicom to png & mha selected array to png (scale *255!!!)
# 5. Blend dicom and mha
# end result 3 images per 

"""
Refined table generation for 2D analysis using PNGs.
"""
import os, time
import itk
import click
import numpy as np
from PIL import Image 
from filehash import FileHash

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

logger = init_logger()
logger.info("Starting data_processing.radiology.refined_table.annotation.generate")


def create_png(src_path, uuid, png_dir):
    """
    Create a png from src_path image, and save a png in png_dir.
    Returns a path to the png file along with an instance number in case of a 3d mha.
    """
    file_path = src_path.split(':')[-1]

    image = itk.imread(file_path)
    arrays = itk.GetArrayViewFromImage(image)

    num_images = arrays.shape[0]
    dicom_idx_in_series = num_images

    if num_images == 1:
        # Convert 2d image to float to avoid overflow or underflow losses.
        image_2d = arrays[0].pixel_array.astype(float)

    else:
        # Find the 2d image with Segmentation from 3d annotation.
        # Some reverse engineering.. save the instance number 
        # from the series to identify the dicom image that was annotated.
        for i,v in enumerate(arrays):
            if np.any(v):
                image_2d = v
                # double check that subtracting is needed for all.
                dicom_idx_in_series -= i+1
                print(i+1)
                return

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Save png named as hash of dicom/mha
    png_path = os.path.join(png_dir, uuid.split("-")[-1])
    im = Image.fromarray(image_2d_scaled)

    if num_images == 1:
        im.save(png_path)
        return png_path
    else:
        # save segmentation in red color.
        rgb = im.convert('RGB')
        red_channel = rgb.getdata(0)
        rgb.putdata(red_channel)
        rgb.save(png_path)
        return dicom_idx_in_series + "|" + png_path

def overlay_pngs(dicom_png_path, mha_png_path, png_dir):
    # blend dicom and segmentation images with 7:3 ratio. 
    dcmpng = Image.open(dicom_png_path).convert('RGB')
    mhapng = Image.open(mha_png_path)

    res = Image.blend(dcmpng, mhapng, 0.3)

    filename = mha_png_path.split("/")[-1]
    filepath = os.path.join(png_dir, "overlay-"+filename)
    res.save(filepath)


def generate_uuid(png_path):

    uuid = FileHash('sha256').hash_file(png_path)

    return "PNG-"+uuid


@click.command()
@click.option('-f', '--config_file', default='config.yaml', required=True, 
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-t', '--data_config_file', default='data_processing/services/config.yaml', required=True,
    help="path to data configuration file. See data_processing/services/config.yaml.template")
def cli(config_file, data_config_file):
    """
    This module takes a SeriesInstanceUID, calls a script to generate volumetric images, and updates the scan table.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.radiology.refined_table.annotation.generate \
	--data_config_file data_processing/services/config.yaml \
	--config_file config.yaml
    """
    start_time = time.time()

    cfg = ConfigSet(name=const.APP_CFG, config_file=config_file)
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

    generate_png_tables(cfg)

    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_png_tables(cfg):
    """
    Create pngs for all dicoms in a series that have corresponding annotations.
    Generate dicom_png and mha_png tables.
    """
    # setup project path
    base_path = cfg.get_value(name=const.DATA_CFG, jsonpath='MIND_DATA_PATH')
    project_path = os.path.join(base_path, cfg.get_value(name=const.DATA_CFG, jsonpath='PROJECT_NAME'))
    logger.info("Got project path : " + project_path)

    # load dicom and mha tables
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='dicom-to-png')

    dicom_table_path = os.path.join(project_path, const.DICOM_TABLE) # earlier version
    mha_table_path = os.path.join(project_path, const.TABLE_DIR, const.TABLE_NAME(cfg)) # MHA_BR_16-512_20201212

    dicom_df = spark.read.format("delta").load(dicom_table_path)
    dicom_df = dicom_df.drop("content")

    mha_df = spark.read.format("delta").load(mha_table_path)
    logger.info("Loaded dicom and mha tables")

    # join with accession number and series number/description
    dicom_alias = dicom_df.select("dicom_record_uuid",
                    dicom_df.path.alias("dicom_path"),
                    dicom_df.metadata.alias("dicom_metadata"))

    cond = [dicom_alias.dicom_metadata.AccessionNumber == mha_df.accession_number, 
            dicom_alias.dicom_metadata.SeriesDescription == 'Ph1/Axial T1 FS post'] # Use series number once available

    subset_df = dicom_alias.join(mha_df, cond)
    logger.info("Joined dicom and mha tables")

    generate_uuid_udf = F.udf(generate_uuid, StringType())

    with CodeTimer(logger, 'Generate pngs and dicom_png table'):

        png_dir = os.path.join(project_path, const.DICOM_PNGS)

        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        dicom_png_table_path = os.path.join(project_path, const.TABLE_DIR, 
            "{0}_{1}".format("DICOM_PNG", cfg.get_value(name=const.DATA_CFG, jsonpath='DATASET_NAME')))

        create_png_udf = F.udf(create_png, StringType())
        subset_df = subset_df.withColumn("dicom_png_path", 
            F.lit(create_png_udf(subset_df.dicom_path, subset_df.dicom_record_uuid,F.lit(png_dir))))

        subset_df = subset_df.withColumn("png_record_uuid",
            F.lit(generate_uuid_udf(subset_df.dicom_png_path)))

        subset_df.show(10, False)

        columns = ["dicom_record_uuid", F.col("dicom_path").alias("path"), "png_record_uuid", "dicom_png_path", F.col("metadata.InstanceNumber").alias("instance_number")]
        subset_df.select(columns) \
            .coalesce(cfg.get_value(name=const.DATA_CFG, jsonpath='NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(dicom_png_table_path)

    with CodeTimer(logger, 'Generate pngs and mha_png table'):

        # mha_png_path: create pngs for all mhas
        png_dir = os.path.join(project_path, const.MHA_PNGS)

        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        mha_png_table_path = os.path.join(project_path, const.TABLE_DIR, 
            "{0}_{1}".format("MHA_PNG", cfg.get_value(name=const.DATA_CFG, jsonpath='DATASET_NAME')))

        mha_df = mha_df.withColumn("instanceid_mha_png_path", 
            F.lit(create_png_udf(mha_df.path, mha_df.scan_annotation_record_uuid, F.lit(png_dir))))

        # split instanceid|mha_png_path into 2 columns
        split_col = F.split(mha_df.instanceid_mha_png_path, '|')
        mha_df = mha_df.withColumn("instance_number", split_col.getItem(0)) \
                    .withColumn("mha_png_path", split_col.getItem(1)) \
                    .drop("instanceid_mha_png_path")

        mha_df = mha_df.withColumn("png_record_uuid",
            F.lit(generate_uuid_udf(mha_df.mha_png_path)))

        # overlay_path: blend mha and "the dicom" found through mha looping
        overlay_pngs_udf = F.udf(overlay_pngs, StringType())

        cond = [subset_df.metadata.AccessionNumber == mha_df.accession_number, 
            subset_df.metadata.SeriesDescription == 'Ph1/Axial T1 FS post',
            subset_df.metadata.InstanceNumber == mha_df.instance_number]
        mha_png_df = mha_df.select(mha_df.png_record_uuid.alias("mha_png_record_uuid"), mha_df.accession_number, mha_df.instance_number) \
            .join(subset_df, cond)

        mha_png_df = mha_png_df.withColumn("overlay_path",
            F.lit(overlay_pngs_udf(mha_png_df.dicom_png_path, mha_png_df.mha_png_path, F.lit(png_dir))))
        
        mha_png_df.show(10, False)
        
        columns = [F.col("mha_png_record_uuid").alias("png_record_uuid"), "mha_png_path", "dicom_png_path", "instance_number", "overlay_path"]
        mha_png_df.select(columns) \
            .coalesce(cfg.get_value(name=const.DATA_CFG, jsonpath='NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(mha_png_table_path)


if __name__ == "__main__":
    cli()

