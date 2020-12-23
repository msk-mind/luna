"""
Generates Refined PNG table for dicom slices in the series that

1) have 3D segmentations
2) match user's SQL where clause (filter based on dicom metadata)

This process uses dicom and mhd annotation proxy tables.

The PNG table contains paths to dicom png and overlay image that combines dicom and its corresponding segmentation.
"""

import os, time, shutil
import click
import numpy as np
from PIL import Image 
from medpy.io import load
from filehash import FileHash

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField

logger = init_logger()
logger.info("Starting data_processing.radiology.refined_table.annotation.generate")


def create_dicom_png(src_path, uuid, accession_number, png_dir):
    """
    Create a png from src_path image, and save a png in png_dir.
    """
    file_path = src_path.split(':')[-1]

    data, header = load(file_path)

    # Convert 2d image to float to avoid overflow or underflow losses.
    # Transpose to get the preserve x, y coordinates.
    image_2d = data[:,:,0].astype(float).T

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Save png named as hash of dicom/seg
    png_path = os.path.join(png_dir, accession_number)
    os.makedirs(png_path, exist_ok=True)

    png_path = os.path.join(png_path, uuid.split("-")[-1] + '.png')

    im = Image.fromarray(image_2d_scaled)
    im.save(png_path)
    
    return png_path
   

def create_seg_png(src_path, uuid, accession_number, png_dir):
    """
    Create pngs from src_path image, and save pngs in png_dir.
    Returns an array of (instance_number, seg_png_path) tuple.
    """
    # Save png named as hash of dicom/seg
    png_dir = os.path.join(png_dir, accession_number)
    os.makedirs(png_dir, exist_ok=True)

    file_path = src_path.split(':')[-1]
    data, header = load(file_path)

    num_images = data.shape[2]

    # Find the annotated slices with 3d segmentation.
    # Some reverse engineering.. save the instance numbers 
    # from the series to identify the dicom slices that were annotated.
    slices = []
    for i in range(num_images):
        image_slice = data[:,:,i]
        if np.any(image_slice):
            image_2d = image_slice.astype(float).T
            # double check that subtracting is needed for all.
            slice_num = num_images - (i+1)

            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            image_2d_scaled = np.uint8(image_2d_scaled)

            png_path = os.path.join(png_dir, "{0}_{1}.png".format(uuid.split("-")[-1], str(slice_num)))

            im = Image.fromarray(image_2d_scaled)

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)
            rgb.save(png_path)

            slices.append( (slice_num, png_path) )

    return slices

 
def overlay_pngs(uuid, instance_number, dicom_path, seg_png_path, accession_number, png_dir):
    """
    Create dicom png images.
    Blend dicom and segmentation images with 7:3 ratio. 
    Returns the path to the combined image.
    """
    
    dicom_png_path = create_dicom_png(dicom_path, uuid, accession_number, png_dir)
    
    dcmpng = Image.open(dicom_png_path).convert('RGB')
    segpng = Image.open(seg_png_path)

    res = Image.blend(dcmpng, segpng, 0.3)

    filename = seg_png_path.split("/")[-1]
    filedir = os.path.join(png_dir, accession_number)

    overlay_png_path = os.path.join(filedir, "overlay_"+filename)
    res.save(overlay_png_path)

    return dicom_png_path + ":" + overlay_png_path


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
    Generate dicom_png and seg_png tables.
    """
    # setup project path
    project_path = os.path.join(cfg.get_value(path=const.DATA_CFG+'::MIND_DATA_PATH'),
                                cfg.get_value(path=const.DATA_CFG+'::PROJECT_NAME'))
    logger.info("Got project path : " + project_path)

    # load dicom and seg tables
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='dicom-to-png')

    # TODO earlier version.. dicom table path can be removed once clean up is done.
    dicom_table_path = cfg.get_value(path=const.DATA_CFG+'::DICOM_TABLE_PATH')
    seg_table_path = os.path.join(project_path, const.TABLE_DIR, 
                     "{0}_{1}".format("MHD", cfg.get_value(path=const.DATA_CFG+'::DATASET_NAME')))

    dicom_df = spark.read.format("delta").load(dicom_table_path)
   
    # subset dicom based on SQL_STRING - to identify a series within the case.
    dicom_df = dicom_df.drop("content") \
                       .filter(cfg.get_value(path=const.DATA_CFG+'::SQL_STRING'))

    seg_df = spark.read.format("delta").load(seg_table_path)
    logger.info("Loaded dicom and seg tables")

    # join with accession number and series number/description
    seg_alias = seg_df.select(seg_df.accession_number,
                    seg_df.path.alias("seg_path"),
                    seg_df.metadata.alias("seg_metadata"))

    cond = [dicom_df.metadata.AccessionNumber == seg_df.accession_number] # Add series number match once available

    subset_df = dicom_df.join(seg_alias, cond)
    logger.info("Joined dicom and seg tables")

    with CodeTimer(logger, 'Generate pngs and seg_png table'):

        # seg_png_path: create pngs for all segs
        png_dir = os.path.join(project_path, const.PNGS)
        seg_png_dir = png_dir + "_seg"

        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(seg_png_dir, exist_ok=True)

        seg_png_table_path = os.path.join(project_path, const.TABLE_DIR, const.TABLE_NAME(cfg))

        create_seg_png_udf = F.udf(create_seg_png, ArrayType(StructType([StructField("instance_number", IntegerType()),
									 StructField("seg_png_path", StringType())])))
        
        seg_df = seg_df.withColumn("slices_pngs", 
            F.lit(create_seg_png_udf("path", "scan_annotation_record_uuid", "accession_number", F.lit(seg_png_dir))))

        logger.info("Created segmentation pngs")

        seg_df = seg_df.withColumn("slices_pngs", F.explode("slices_pngs")) \
                       .select(F.col("slices_pngs.instance_number").alias("instance_number"), F.col("slices_pngs.seg_png_path").alias("seg_png_path"),
                               "scan_annotation_record_uuid", "accession_number", "path") 

        logger.info("Exploded rows")
	
        # overlay_path: blend seg and the dicom instance
        overlay_png_udf = F.udf(overlay_pngs, StringType())

        seg_df = seg_df.select(seg_df.accession_number.alias("access_no"), seg_df.path.alias("seg_path"),
                               "instance_number", "seg_png_path", "scan_annotation_record_uuid")
        
        cond = [subset_df.metadata.AccessionNumber == seg_df.access_no, subset_df.metadata.InstanceNumber == seg_df.instance_number] 
        
        seg_df = seg_df.join(subset_df, cond)

        seg_df = seg_df.withColumn("dicom_overlay",
            F.lit(overlay_png_udf("dicom_record_uuid", "instance_number", "path", "seg_png_path", "accession_number", F.lit(png_dir))))
 
        # split dicom:overlay paths
        split_col = F.split(seg_df.dicom_overlay, ':')
        seg_df = seg_df.withColumn("dicom_png_path", split_col.getItem(0)) \
                    .withColumn("overlay_path", split_col.getItem(1)) \
                    .drop("dicom_overlay")

        generate_uuid_udf = F.udf(generate_uuid, StringType())
        seg_df = seg_df.withColumn("png_record_uuid", F.lit(generate_uuid_udf(seg_df.overlay_path)))
       
        logger.info("Created overlay images")
        
        columns = ["png_record_uuid", "metadata", "dicom_png_path", "overlay_path", "scan_annotation_record_uuid"]

        seg_df.select(columns) \
            .coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(seg_png_table_path)

        # clean up seg png folder
        shutil.rmtree(seg_png_dir)
       
if __name__ == "__main__":
    cli()

