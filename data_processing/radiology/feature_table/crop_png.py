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

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField

logger = init_logger()
logger.info("Starting data_processing.radiology.feature_table.crop_png")


def generate_uuid(path):

    uuid = FileHash('sha256').hash_file(path)

    return "FEATURE-"+uuid


@click.command()
@click.option('-f', '--config_file', default='config.yaml', required=True, 
    help="path to config file containing application configuration. See config.yaml.template")
@click.option('-t', '--data_config_file', default='data_processing/radiology/feature_table/config.yaml', required=True,
    help="path to data configuration file. See data_processing/radiology/feature_table/config.yaml.template")
def cli(config_file, data_config_file):
    """
    This module takes a SeriesInstanceUID, calls a script to generate volumetric images, and updates the scan table.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.radiology.refined_table.annotation.generate \
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
    Create pngs for all dicoms in a series that have corresponding annotations.
    Generate dicom_png and seg_png tables.
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
    mha_df = spark.read.format("delta").load(mha_table_path)
 
    logger.info("Loaded mha and png tables")

    df = png_df.join(mha_df, "scan_annotation_record_uuid")
    logger.info("Joined mha and png tables")
    logger.info("COUNT::: " + str(df.count()))

    with CodeTimer(logger, 'Crop pngs and create feature table'):

        # seg_png_path: create pngs for all segs
        features_dir = os.path.join(project_path, const.FEATURES)

        os.makedirs(features_dir, exist_ok=True)

        feature_table_path = os.path.join(project_path, const.TABLE_DIR, "{0}_{1}".format("FEATURES", DATASET_NAME))
	
        def crop_pngs(df: pd.DataFrame) -> pd.DataFrame:
		    """
		    Find the centroid of the 2d segmentation for the scan.
		    Crop PNG images around the centroid.
		    """
		    # Save feature pngs
		    features = os.path.join(feature_dir, df.metadata.get("AccessionNumber"))
		    os.makedirs(features, exist_ok=True)

		    # mha file path
		    file_path = df.path.split(':')[-1]
		    data, header = load(file_path)

		    h, w, num_images = data.shape

		    # Find the annotated slice
		    xcenter, ycenter = 0, 0
		    for i in range(num_images):
		        image_slice = data[:,:,i]
		        if np.any(image_slice):
		            image_slice = image_slice.astype(float)
		            
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

		    # Crop all overlay, dicom pngs in the scan.
		    # dicom_pngs and overlay_pngs are in the same directory pngs/accession_number
		    png_dir = os.path.dirname(df.dicom_png_path)

		    pngs = glob.glob(os.path.join(png_dir, "*.png"))

		    for png in pngs:

		        basename = os.path.basename(png)
		        cropped_file = os.path.join(features, basename)

		        im = Image.open(png)
		        im_crop = im.crop((xmin, ymin, xmax, ymax))
		        im_crop.save(cropped_file)

		        if basename.startswith("overlay"):
		        	df["overlay_path"] = cropped_file
		        else:
		        	df["dicom_png_path"] = cropped_file

		    return df
        
        df = df.groupBy("scan_annotation_record_uuid").applyInPandas(crop_png, schema = df.schema)
       
        logger.info("Cropped pngs")

        generate_uuid_udf = F.udf(generate_uuid, StringType())
        df = df.withColumn("feature_record_uuid", F.lit(generate_uuid_udf(df.overlay_path)))

        columns = ["feature_record_uuid", "metadata", "dicom_png_path", "overlay_path", "png_record_uuid"]

        df.select(columns) \
            .coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta") \
            .mode("overwrite") \
            .save(feature_table_path)


if __name__ == "__main__":
    cli()

