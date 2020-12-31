"""
This final preprocessing step
1. Loads PNG table and MHA table (2d segmentation)
2. Finds the centroid of the 2d segmentation for the scan.
3. Crop PNG images around the centroid.
4. Saves the raw PNG images and paths in a table.
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
logger.info("Starting data_processing.radiology.feature_table.generate")


def generate_uuid(content):
    content = BytesIO(content)

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        uuid = FileHash('sha256').hash_file(content)
 
    #uuid = FileHash('sha256').hash_file(path)
    return "FEATURE-"+uuid

def crop_and_save(png, features, xmin, ymin, xmax, ymax):
    # crop png with xmin, ymin, xmax, ymax boundary
    # save cropped image in in features directory
    png_name = os.path.basename(png)
    feature_path = os.path.join(features, png_name)

    im = Image.open(png)
    im_crop = im.crop((xmin, ymin, xmax, ymax))
    im_crop.save(feature_path)
    return feature_path
    #return im_crop.tobytes()


@click.command()
@click.option('-f', '--config_file', default='config.yaml', required=True, 
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-t', '--data_config_file', default='data_processing/radiology/feature_table/config.yaml', required=True,
    help="path to data configuration file. See data_processing/radiology/feature_table/config.yaml.template")
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
    CROP_SIZE = cfg.get_value(path=const.DATA_CFG+'::CROP_SIZE')

    png_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("PNG", DATASET_NAME))
    mha_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("MHA", DATASET_NAME))

    png_df = spark.read.format("delta").load(png_table_path)
    # TODO filtering here can be removed once we have more metadata for the annotations
    png_df = png_df.filter(cfg.get_value(path=const.DATA_CFG+'::SQL_STRING'))
    mha_df = spark.read.format("delta").load(mha_table_path) \
                  .drop("metadata", "scan_annotation_record_uuid")
 
    logger.info("Loaded mha and png tables")

    columns = [F.col("metadata.InstanceNumber").alias("instance_number").cast(IntegerType()), "accession_number", 
               "dicom_path", "overlay_path", "path", "png_record_uuid"]

    df = mha_df.join(png_df, png_df.metadata.AccessionNumber == mha_df.accession_number) \
               .drop("scan_annotation_record_uuid") \
               .select(columns).distinct() \
               .dropna(subset=["dicom_path", "overlay_path"])

    logger.info("Joined mha and png tables")
    logger.info("COUNT::: " + str(df.count()))

    with CodeTimer(logger, 'Crop pngs and create feature table'):

        # seg_png_path: create pngs for all segs
        feature_dir = os.path.join(project_path, const.FEATURES)

        feature_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("FEATURE", DATASET_NAME))
    
        def crop_images(df: pd.DataFrame) -> pd.DataFrame:
            """
            Find the centroid of the 2d segmentation for the scan.
            Crop PNG images around the centroid.
            """
            # Save feature pngs
            print("Processing accession number: " + str(df.accession_number.values[0]))
            #features = os.path.join(feature_dir, df.accession_number.values[0])
            #os.makedirs(features, exist_ok=True)

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
            rad = CROP_SIZE // 2
            xmin, ymin, xmax, ymax = (xcenter - rad), (ycenter - rad), (xcenter + rad), (ycenter + rad)

            if xmin < 0:
                xmin = 0
                xmax = CROP_SIZE

            if xmax > w:
                xmin = w - CROP_SIZE
                xmax = w

            if ymin < 0:
                ymin = 0
                ymax = CROP_SIZE

            if ymax > h:
                ymin = h - CROP_SIZE
                ymax = h

            # Crop all overlay, dicom pngs.
            dicom_paths = []
            for png in df.dicom_path.values:
                #feature_path = crop_and_save(png, features, xmin, ymin, xmax, ymax)
                im = Image.open(png)
                feature_path = im.crop((xmin, ymin, xmax, ymax)).tobytes()
                dicom_paths.append(feature_path)

            overlay_paths = []
            for png in df.overlay_path.values:
                #feature_path = crop_and_save(png, features, xmin, ymin, xmax, ymax)
                im = Image.open(png)
                feature_path = im.crop((xmin, ymin, xmax, ymax)).tobytes()
                overlay_paths.append(feature_path)

            df["dicom_path"] = dicom_paths
            df["overlay_path"] = overlay_paths
            
            return df
        
        logger.info(df.schema)
        schema = StructType([StructField("accession_number",StringType(),True),
                             StructField("instance_number",IntegerType(),True),
                             StructField("dicom_path",BinaryType(),True),
                             StructField("overlay_path",BinaryType(),True),
                             StructField("png_record_uuid",StringType(),True),
                             StructField("path",StringType(),True)])
        df = df.groupBy("accession_number").applyInPandas(crop_images, schema = schema)
       
        logger.info("Cropped pngs")

        spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
        generate_uuid_udf = F.udf(generate_uuid, StringType())
        df = df.withColumn("feature_record_uuid", F.lit(generate_uuid_udf(df.overlay_path)))

        columns = ["feature_record_uuid", "accession_number", "instance_number", "dicom_path", "overlay_path", "png_record_uuid"]

        df.select(columns) \
            .coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(feature_table_path)


if __name__ == "__main__":
    cli()

