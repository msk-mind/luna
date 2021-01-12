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
import pandas as pd
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
    project_path = os.path.join(cfg.get_value(path=const.DATA_CFG+'::MIND_DATA_PATH'),
                                cfg.get_value(path=const.DATA_CFG+'::PROJECT_NAME'))

    # load dicom and seg tables
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='crop_png')

    DATASET_NAME = cfg.get_value(path=const.DATA_CFG+'::DATASET_NAME')
    CROP_WIDTH = int(cfg.get_value(path=const.DATA_CFG+'::CROP_WIDTH'))
    CROP_HEIGHT = int(cfg.get_value(path=const.DATA_CFG+'::CROP_HEIGHT'))
    IMAGE_WIDTH = int(cfg.get_value(path=const.DATA_CFG+'::IMAGE_WIDTH'))
    IMAGE_HEIGHT = int(cfg.get_value(path=const.DATA_CFG+'::IMAGE_HEIGHT'))

    png_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("PNG", DATASET_NAME))
    mha_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("MHA", DATASET_NAME))

    png_df = spark.read.format("delta").load(png_table_path)
    # TODO filtering here can be removed once we have more metadata for the annotations
    png_df = png_df.filter(cfg.get_value(path=const.DATA_CFG+'::SQL_STRING'))
    mha_df = spark.read.format("delta").load(mha_table_path) \
                  .drop("metadata", "scan_annotation_record_uuid")
 
    logger.info("Loaded mha and png tables")

    columns = [F.col("metadata.InstanceNumber").alias("instance_number").cast(IntegerType()), 
                F.col("metadata.PatientID").alias("xnat_patient_id"), 
                F.col("metadata.SeriesInstanceUID").alias("SeriesInstanceUID"),
                "accession_number", "dicom", "overlay", "path", "png_record_uuid", "scan_annotation_record_uuid"]

    df = png_df.join(mha_df, png_df.metadata.AccessionNumber == mha_df.accession_number) \
               .select(columns).distinct() \
               .dropna(subset=["dicom", "overlay"])

    logger.info("Joined mha and png tables")

    with CodeTimer(logger, 'Crop pngs and create feature table'):

        feature_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("FEATURE", DATASET_NAME))
    
        def crop_images(df: pd.DataFrame) -> pd.DataFrame:
            """
            Find the centroid of the 2d segmentation for the scan.
            Crop PNG images around the centroid.
            """
            print("Processing accession number: " + str(df.accession_number.values[0]))
           
            scan_uuid, dicom_features, overlay_features = [], [], []
            
            # mha file path
            file_path = df.path.values[0].split(':')[-1]
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

            # Find xmin, ymin, xmax, ymax based on CROP_SIZE
            width_rad = CROP_WIDTH // 2
            height_rad = CROP_HEIGHT // 2
            # Check if h,w matches IMAGE_WIDTH, IMAGE_HEIGHT. If not, this is due to png being rescaled. So scale centers.
            if not h == IMAGE_HEIGHT and not w == IMAGE_WIDTH:
                xcenter = int(xcenter * IMAGE_WIDTH//w)
                ycenter = int(ycenter * IMAGE_HEIGHT//h)

            xmin, ymin, xmax, ymax = (xcenter - width_rad), (ycenter - height_rad), (xcenter + width_rad), (ycenter + height_rad)

            if xmin < 0:
                xmin = 0
                xmax = CROP_WIDTH

            if xmax > IMAGE_WIDTH:
                xmin = IMAGE_WIDTH - CROP_WIDTH
                xmax = IMAGE_WIDTH

            if ymin < 0:
                ymin = 0
                ymax = CROP_HEIGHT

            if ymax > IMAGE_HEIGHT:
                ymin = IMAGE_HEIGHT - CROP_HEIGHT
                ymax = IMAGE_HEIGHT

            # Crop overlay, dicom pngs.
            for png in df.dicom.values:
                im = Image.frombytes("L", (IMAGE_WIDTH, IMAGE_HEIGHT), bytes(png)) 
                feature = im.crop((xmin, ymin, xmax, ymax)).tobytes()
                dicom_features.append(feature)

            for png in df.overlay.values:
                im = Image.frombytes("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), bytes(png)) 
                feature = im.crop((xmin, ymin, xmax, ymax)).tobytes()
                overlay_features.append(feature)

            scan_uuid.extend(df.scan_annotation_record_uuid.values)

            df["dicom"] = dicom_features
            df["overlay"] = overlay_features
            df["scan_annotation_record_uuid"] = scan_uuid
            
            return df
        
        schema = StructType([StructField("xnat_patient_id",StringType(),True),
                             StructField("accession_number",StringType(),True),
                             StructField("instance_number",IntegerType(),True),
                             StructField("SeriesInstanceUID",StringType(),True),
                             StructField("dicom",BinaryType(),True),
                             StructField("overlay",BinaryType(),True),
                             StructField("png_record_uuid",StringType(),True),
                             StructField("scan_annotation_record_uuid",StringType(),True),
                             StructField("path",StringType(),True)])
        #df = df.groupBy("accession_number").applyInPandas(crop_images, schema = schema)
        df = df.groupBy("accession_number", "scan_annotation_record_uuid").applyInPandas(crop_images, schema = schema)
       
        logger.info("Cropped pngs")

        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        generate_uuid_udf = F.udf(generate_uuid, StringType())
        df = df.withColumn("feature_record_uuid", F.lit(generate_uuid_udf(df.overlay)))

        columns = ["feature_record_uuid", "accession_number", "instance_number", "SeriesInstanceUID", 
                    "xnat_patient_id", "dicom", "overlay", "png_record_uuid","scan_annotation_record_uuid"]

        df.select(columns).distinct() \
            .coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')) \
            .write.format("delta") \
            .mode("overwrite") \
            .save(feature_table_path)


if __name__ == "__main__":
    cli()

