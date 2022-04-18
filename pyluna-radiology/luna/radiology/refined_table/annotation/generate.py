"""
Generates Refined PNG table for dicom slices in the series that have 3D segmentations

This process uses dicom and mhd annotation proxy tables.

The PNG table contains paths to dicom png and overlay image that combines dicom and its corresponding segmentation.
"""

import os, time, shutil
import click

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.common.custom_logger import init_logger
import luna.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField, BinaryType

from luna.common.utils import get_absolute_path

logger = init_logger()
logger.info("Starting luna.radiology.refined_table.annotation.generate")


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
            python3 -m luna.radiology.refined_table.annotation.generate \
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

    scan_table_path = cfg.get_value(path=const.DATA_CFG + '::SCAN_TABLE_PATH')
    seg_table_path = cfg.get_value(path=const.DATA_CFG + '::LABEL_TABLE_PATH')

    scan_df = spark.read.format("delta").load(scan_table_path) \
                   .select(F.col("metadata").alias("scan_metadata"), F.col("path").alias("scan_path"),
                                            F.col("subset_path").alias("subset_scan_path"))

    seg_df = spark.read.format("delta").load(seg_table_path)
    logger.info("Loaded dicom and seg tables")
    
    with CodeTimer(logger, 'Generate pngs and seg_png table'):

        width = cfg.get_value(path=const.DATA_CFG+'::IMAGE_WIDTH')
        height = cfg.get_value(path=const.DATA_CFG+'::IMAGE_HEIGHT')
        n_slices = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::N_SLICES'))
        crop_width = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::CROP_WIDTH'))
        crop_height = validate_integer_param(cfg.get_value(path=const.DATA_CFG+'::CROP_HEIGHT'))

        # optional columns from radiology annotation table to include.
        metadata_columns = []
        if cfg.has_value(path=const.DATA_CFG+'::METADATA_COLUMNS'):
            metadata_columns = cfg.get_value(path=const.DATA_CFG+'::METADATA_COLUMNS')
        seg_png_table_path = const.TABLE_LOCATION(cfg)

        # join scan and seg tables
        seg_df = seg_df.join(scan_df,
                             scan_df.scan_metadata.SeriesInstanceUID == seg_df.series_instance_uid)
        logger.info(seg_df.count())

        # find images with tumor
        spark.sparkContext.addPyFile(get_absolute_path(__file__, "../../common/preprocess.py"))
        from preprocess import create_images
        create_images_udf = F.udf(create_images, ArrayType(StructType(
                                    [StructField("n_tumor_slices", IntegerType()),
                                     StructField("dicom", BinaryType()),
                                     StructField("overlay", BinaryType())])))

        seg_df = seg_df.withColumn("slices_images",
            F.lit(create_images_udf("scan_path", "path", "subset_scan_path", "subset_path",
                                     F.lit(width), F.lit(height), F.lit(crop_width), F.lit(crop_height), F.lit(n_slices))))

        logger.info("Created pngs")

        # add metadata_columns
        columns_to_select = [F.col("slices_images.n_tumor_slices").alias("n_tumor_slices"),
                             F.col("slices_images.dicom").alias("dicom"),
                             F.col("slices_images.overlay").alias("overlay"),
                             F.col("scan_metadata").alias("metadata"), 
                             "accession_number", "series_number", "path", "subset_path", "scan_annotation_record_uuid"]
        columns_to_select.extend(metadata_columns)

        seg_df = seg_df.withColumn("slices_images", F.explode("slices_images")) \
                       .select(columns_to_select)
        logger.info("Exploded rows")

        # generate uuid
        spark.sparkContext.addPyFile(
            get_absolute_path(__file__, "../../../../../pyluna-common/luna/common/utils.py"))
        from utils import generate_uuid_binary
        generate_uuid_udf = F.udf(generate_uuid_binary, StringType())
        seg_df = seg_df.withColumn("png_record_uuid", F.lit(generate_uuid_udf(seg_df.overlay, F.array([F.lit("PNG")]))))

        seg_df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(seg_png_table_path)

       
if __name__ == "__main__":
    cli()

