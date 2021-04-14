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
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField, BinaryType

logger = init_logger()
logger.info("Starting data_processing.radiology.refined_table.annotation.generate")


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
        This module generates a delta table with image and scan radiology data based on the input and output parameters        specified in the data_config_file.

        Example:
            python3 -m data_processing.radiology.refined_table.annotation.generate \
                     --data_config_file <path to data config file> \
                     --app_config_file <path to app config file>
    """
    start_time = time.time()

    cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
    cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

    # copy app and data configuration to destination config dir
    config_location = const.CONFIG_LOCATION(cfg)
    os.makedirs(config_location, exist_ok=True)

    shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
    shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
    logger.info("config files copied to %s", config_location)

    generate_image_table()

    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def validate_integer_param(param):
    """
    if param is not empty and numeric, return integer. otherwise return an empty string
    """
    if str(param).isnumeric():
        param = int(param)
    else:
        param = ""
    return param


def generate_image_table():
    """
    Create pngs for all dicoms in a series that have corresponding annotations.
    Generate dicom_png and seg_png tables.
    """
    cfg = ConfigSet()

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

    seg_df = spark.read.format("delta").load(seg_table_path)
    logger.info("Loaded dicom and seg tables")
    
    with CodeTimer(logger, 'Generate pngs and seg_png table'):

        width = cfg.get_value(path=const.DATA_CFG+'::IMAGE_WIDTH')
        height = cfg.get_value(path=const.DATA_CFG+'::IMAGE_HEIGHT')
        n_slices = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::N_SLICES'))
        crop_width = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::CROP_WIDTH'))
        crop_height = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::CROP_HEIGHT'))

        seg_png_table_path = const.TABLE_LOCATION(cfg)
        
        # find images with tumor
        spark.sparkContext.addPyFile("./data_processing/radiology/common/preprocess.py")
        from preprocess import create_seg_images, overlay_images
        create_seg_png_udf = F.udf(create_seg_images, ArrayType(StructType(
                                    [StructField("instance_number", IntegerType()),
                                     StructField("scan_annotation_record_uuid", StringType()),
                                     StructField("seg_png", BinaryType()),
                                     StructField("x", IntegerType()),
                                     StructField("y", IntegerType())])))

        seg_df = seg_df.withColumn("slices_uuid_pngs_xy",
            F.lit(create_seg_png_udf("path", "scan_annotation_record_uuid", F.lit(width), F.lit(height), F.lit(n_slices))))

        logger.info("Created segmentation pngs")

        seg_df = seg_df.withColumn("slices_uuid_pngs_xy", F.explode("slices_uuid_pngs_xy")) \
                       .select(F.col("slices_uuid_pngs_xy.instance_number").alias("instance_number"),
                               F.col("slices_uuid_pngs_xy.seg_png").alias("seg_png"),
                               F.col("slices_uuid_pngs_xy.scan_annotation_record_uuid").alias("scan_annotation_record_uuid"),
                               F.col("slices_uuid_pngs_xy.x").alias("x_center"),
                               F.col("slices_uuid_pngs_xy.y").alias("y_center"),
                               "accession_number", "series_number", "path", "label")

        logger.info("Exploded rows")

        # create overlay images: blend seg and the dicom images
        seg_df = seg_df.select("accession_number", seg_df.path.alias("seg_path"), "label",
                               "instance_number", "seg_png", "scan_annotation_record_uuid", "series_number",
                               "x_center", "y_center")

        cond = [dicom_df.metadata.AccessionNumber == seg_df.accession_number,
                dicom_df.metadata.SeriesNumber == seg_df.series_number,
                dicom_df.metadata.InstanceNumber == seg_df.instance_number]

        seg_df = seg_df.join(dicom_df, cond)

        overlay_image_udf = F.udf(overlay_images, StructType([StructField("dicom", BinaryType()),
                                                              StructField("overlay", BinaryType())]))

        seg_df = seg_df.withColumn("dicom_overlay",
            F.lit(overlay_image_udf("path", "seg_png", F.lit(width), F.lit(height), "x_center", "y_center",
                                    F.lit(crop_width), F.lit(crop_height))))
 
        # unpack dicom_overlay struct into 2 columns
        seg_df = seg_df.select(F.col("dicom_overlay.dicom").alias("dicom"), F.col("dicom_overlay.overlay").alias("overlay"),
                                "metadata", "scan_annotation_record_uuid", "label")

        # generate uuid
        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        spark.sparkContext.addPyFile("./data_processing/common/utils.py")
        from utils import generate_uuid_binary
        generate_uuid_udf = F.udf(generate_uuid_binary, StringType())
        seg_df = seg_df.withColumn("png_record_uuid", F.lit(generate_uuid_udf(seg_df.overlay, F.array([F.lit("PNG")]))))
       
        logger.info("Created overlay images")
        
        seg_df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(seg_png_table_path)

       
if __name__ == "__main__":
    cli()

