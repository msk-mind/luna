"""
This final preprocessing step
1. Loads PNG table and MHA table (2d segmentation)
2. Finds the centroid of the 2d segmentation for the scan.
3. Crop PNG images around the centroid.
4. Saves the PNG binaries in a table.
"""
import os, time, glob
import click


from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
from data_processing.common.utils import generate_uuid_binary
import data_processing.common.constants as const
from data_processing.radiology.common.preprocess import find_centroid, crop_images

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, BinaryType

logger = init_logger()
logger.info("Starting data_processing.radiology.feature_table.annotation.generate")


@click.command()
@click.option('-a', '--app_config_file', default='config.yaml', required=True,
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-d', '--data_config_file', default='data_processing/radiology/feature_table/annotation/config.yaml', required=True,
    help="path to data configuration file. See data_processing/radiology/feature_table/annotation/data_config.yaml.template")
def cli(app_config_file, data_config_file):
    """
    This module generates cropped png images based on png and mha (2d segmentation) tables.
 
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.radiology.feature_table.annotation.generate \
        --data_config_file data_processing/radiology/feature_table/config.yaml \
        --app_config_file config.yaml
    """
    start_time = time.time()

    cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
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
    spark.sparkContext.addPyFile("./data_processing/radiology/common/preprocess.py")
    from preprocess import find_centroid, crop_images
    find_centroid_udf = F.udf(find_centroid, StructType([StructField("x", IntegerType()), StructField("y", IntegerType())]))
    mha_df = mha_df.withColumn("center", find_centroid_udf("path", F.lit(IMAGE_WIDTH), F.lit(IMAGE_HEIGHT))) \
                   .select(F.col("center.x").alias("x"), F.col("center.y").alias("y"), "accession_number", "series_number", "scan_annotation_record_uuid", F.col("label").alias("mha_label"))
    
    logger.info("Loaded mha and png tables")

    # Join PNG and MHA tables
    columns = ["metadata", "dicom", "overlay", "png_record_uuid", "scan_annotation_record_uuid", "x","y", "label"]
    
    cond = [png_df.metadata.AccessionNumber == mha_df.accession_number,
            png_df.metadata.SeriesNumber == mha_df.series_number,
            png_df.label.eqNullSafe(mha_df.mha_label)]

    df = mha_df.join(png_df, cond) \
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
        spark.sparkContext.addPyFile("./data_processing/common/utils.py")
        from utils import generate_uuid_binary
        generate_uuid_udf = F.udf(generate_uuid_binary, StringType())
        df = df.withColumn("feature_record_uuid", F.lit(generate_uuid_udf("overlay", F.array([F.lit("FEATURE")]))))

        df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')) \
            .write.format("delta") \
            .mode("overwrite") \
            .save(feature_table_path)
    

if __name__ == "__main__":
    cli()

