"""
Generates Refined PNG table for dicom slices in the series that

1) have 3D segmentations
2) match user's SQL where clause (filter based on dicom metadata)

This process uses dicom and mhd annotation proxy tables.

The PNG table contains paths to dicom png and overlay image that combines dicom and its corresponding segmentation.
"""

import os, time, shutil
import click

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.utils import generate_uuid_binary
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const
from data_processing.radiology.common.utils import overlay_images, create_seg_images

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField, BinaryType

logger = init_logger()
logger.info("Starting data_processing.radiology.refined_table.annotation.generate")


@click.command()
@click.option('-f', '--config_file', default='config.yaml', required=True, 
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-t', '--data_config_file', default='data_processing/refined_table/annotation/config.yaml', required=True,
    help="path to data configuration file. See data_processing/refined_table/annotation/config.yaml.template")
def cli(config_file, data_config_file):
    """
    This module takes a SeriesInstanceUID, calls a script to generate volumetric images, and updates the scan table.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.radiology.refined_table.annotation.generate \
	--data_config_file data_processing/refined_table/annotation/config.yaml \
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
    project_path = const.PROJECT_LOCATION(cfg)
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
    
    with CodeTimer(logger, 'Generate pngs and seg_png table'):

        width = cfg.get_value(path=const.DATA_CFG+'::IMAGE_WIDTH')
        height = cfg.get_value(path=const.DATA_CFG+'::IMAGE_HEIGHT')

        seg_png_table_path = const.TABLE_LOCATION(cfg)
        
        # find images with tumor
        create_seg_png_udf = F.udf(create_seg_images, ArrayType(StructType(
                                    [StructField("instance_number", IntegerType()),
                                     StructField("scan_annotation_record_uuid", StringType()),
				     StructField("seg_png", BinaryType())])))
        
        seg_df = seg_df.withColumn("slices_uuid_pngs", 
            F.lit(create_seg_png_udf("path", "scan_annotation_record_uuid", F.lit(width), F.lit(height))))

        logger.info("Created segmentation pngs")

        seg_df = seg_df.withColumn("slices_uuid_pngs", F.explode("slices_uuid_pngs")) \
                       .select(F.col("slices_uuid_pngs.instance_number").alias("instance_number"), F.col("slices_uuid_pngs.seg_png").alias("seg_png"),
                                F.col("slices_uuid_pngs.scan_annotation_record_uuid").alias("scan_annotation_record_uuid"),
                               "accession_number", "path", "label") 

        logger.info("Exploded rows")

        # create overlay images: blend seg and the dicom images
        seg_df = seg_df.select("accession_number", seg_df.path.alias("seg_path"), "label",
                               "instance_number", "seg_png", "scan_annotation_record_uuid")
        
        cond = [dicom_df.metadata.AccessionNumber == seg_df.accession_number, dicom_df.metadata.InstanceNumber == seg_df.instance_number] 
        
        seg_df = seg_df.join(dicom_df, cond)

        overlay_image_udf = F.udf(overlay_images, StructType([StructField("dicom", BinaryType()), StructField("overlay", BinaryType())]))

        seg_df = seg_df.withColumn("dicom_overlay",
            F.lit(overlay_image_udf("path", "seg_png", F.lit(width), F.lit(height))))
 
        # unpack dicom_overlay struct into 2 columns
        seg_df = seg_df.select(F.col("dicom_overlay.dicom").alias("dicom"), F.col("dicom_overlay.overlay").alias("overlay"),
                                "metadata", "scan_annotation_record_uuid", "label")

        # generate uuid
        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        generate_uuid_udf = F.udf(generate_uuid_binary, StringType())
        seg_df = seg_df.withColumn("png_record_uuid", F.lit(generate_uuid_udf(seg_df.overlay, F.lit("PNG-"))))
       
        logger.info("Created overlay images")
        
        seg_df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(seg_png_table_path)

       
if __name__ == "__main__":
    cli()

