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
from io import BytesIO

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField, BinaryType

logger = init_logger()
logger.info("Starting data_processing.radiology.refined_table.annotation.generate")


def create_dicom_png(src_path, accession_number):
    """
    Create a png from src_path image, and return image as bytes.
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

    print(image_2d_scaled.shape)
    im = Image.fromarray(image_2d_scaled, mode="L")

    return im.tobytes()
       

def create_seg_png(src_path, accession_number):
    """
    Create pngs from src_path image.
    Returns an array of (instance_number, png binary) tuple.
    """
    # Save png named as hash of dicom/seg
    #png_dir = os.path.join(png_dir, accession_number)
    #os.makedirs(png_dir, exist_ok=True)

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

            im = Image.fromarray(image_2d_scaled)

            # save segmentation in red color.
            rgb = im.convert('RGB')
            red_channel = rgb.getdata(0)
            rgb.putdata(red_channel)
            png_binary = rgb.tobytes()

            slices.append( (slice_num, png_binary) )

    return slices

 
def overlay_pngs(instance_number, dicom_path, seg, accession_number, image_size):
    """
    Create dicom png images.
    Blend dicom and segmentation images with 7:3 ratio. 
    Returns tuple of binaries to the combined image.
    """
    dicom_binary = create_dicom_png(dicom_path, accession_number)

    # load dicom and seg images from bytes
    size =  int(image_size)
    dcm_img = Image.frombytes("L", (size, size), bytes(dicom_binary))
    dcm_img = dcm_img.convert("RGB")
    seg_img = Image.frombytes("RGB", (size, size), bytes(seg))

    res = Image.blend(dcm_img, seg_img, 0.3)
    overlay = res.tobytes()

    return (dicom_binary, overlay)


def generate_uuid(content):

    content = BytesIO(content)

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        uuid = FileHash('sha256').hash_file(content)

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
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

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

        seg_png_table_path = os.path.join(project_path, const.TABLE_DIR, const.TABLE_NAME(cfg))
        
        # find images with tumor
        create_seg_png_udf = F.udf(create_seg_png, ArrayType(StructType([StructField("instance_number", IntegerType()),
									 StructField("seg_png", BinaryType())])))
        
        seg_df = seg_df.withColumn("slices_pngs", 
            F.lit(create_seg_png_udf("path", "accession_number")))

        logger.info("Created segmentation pngs")

        seg_df = seg_df.withColumn("slices_pngs", F.explode("slices_pngs")) \
                       .select(F.col("slices_pngs.instance_number").alias("instance_number"), F.col("slices_pngs.seg_png").alias("seg_png"),
                               "accession_number", "path") 

        logger.info("Exploded rows")

        # create overlay images: blend seg and the dicom images
        seg_df = seg_df.select(seg_df.accession_number.alias("access_no"), seg_df.path.alias("seg_path"),
                               "instance_number", "seg_png", "scan_annotation_record_uuid")
        
        cond = [subset_df.metadata.AccessionNumber == seg_df.access_no, subset_df.metadata.InstanceNumber == seg_df.instance_number] 
        
        seg_df = seg_df.join(subset_df, cond)

        overlay_png_udf = F.udf(overlay_pngs, StructType([StructField("dicom", BinaryType()), StructField("overlay", BinaryType())]))

        seg_df = seg_df.withColumn("dicom_overlay",
            F.lit(overlay_png_udf("instance_number", "path", "seg_png", "accession_number", 
                                    F.lit(cfg.get_value(path=const.DATA_CFG+'::IMAGE_SIZE')))))
 
        # unpack dicom_overlay struct into 2 columns
        seg_df = seg_df.select(F.col("dicom_overlay.dicom").alias("dicom"), F.col("dicom_overlay.overlay").alias("overlay"),
                                "metadata", "scan_annotation_record_uuid")

        # generate uuid
        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        generate_uuid_udf = F.udf(generate_uuid, StringType())
        seg_df = seg_df.withColumn("png_record_uuid", F.lit(generate_uuid_udf(seg_df.overlay)))
       
        logger.info("Created overlay images")
        
        seg_df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(seg_png_table_path)

        seg_df.printSchema()
        
       
if __name__ == "__main__":
    cli()

