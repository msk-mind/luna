"""
This module pre-processes the CE-CT acquisitions and associated segmentations and generates
a DataFrame tracking the file paths of the pre-processed items, stored as NumPy ndarrays.

Usage: 
    $ python preprocess_feature.py --spark_master_uri {spark_master_uri} --base_directory {directory/to/tables} --target_spacing {x_spacing} {y_spacing} {z_spacing}  --query "{sql where clause}" --feature_table_output_name {name-of-table-to-be-created}
    
Parameters: 
    REQUIRED PARAMETERS:
        --base_directory: parent directory containing /tables directory, where {base_directory}/tables/scan and {base_directory}/tables/annotation are
                            the scan and annotation delta tables
        --target_spacing: target spacing for the x,y,and z dimensions
        --spark_master_uri: spark master uri e.g. spark://master-ip:7077 or local[*]
    OPTIONAL PARAMETERS:
        --query: where clause of SQL query to filter feature tablE. WHERE does not need to be included, make sure to wrap with quotes to be interpretted correctly
            - Queriable Columns to Filter By:
                - SeriesInstanceUID,AccessionNumber,ct_dates,ct_accession,img,ring-seg,annotation_uid,0_1,vendor,ST,kvP,mA,ID,Rad_segm,R_ovary,L_ovary,Omentum,Notes,subtype,seg
            - examples:
                - filtering by subtype: --query "subtype='BRCA1' or subtype='BRCA2'"
                - filtering by AccessionID: --query "AccessionNumber = '12345'"
        --feature_table_output_name: name of feature table that is created, default is feature-table,
                feature table will be created at {base_directory}/tables/features/{feature_table_output_name}
        --hdfs: base directory is on hdfs or local filesystem
Example:
    $ python preprocess_feature.py --spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables/ --target_spacing 1.0 1.0 3.0  --query "subtype='BRCA1' or subtype='BRCA2'" --feature_table_output_name brca-feature-table
"""
import os, sys, subprocess, time
import click
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../') ))

from sparksession import SparkConfig
from custom_logger import init_logger

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from skimage.transform import resize

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, expr
from pyspark.sql.types import StringType

logger = init_logger()

def process_patient_pandas_udf(patient):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    # TODO add multiprocess logging if spark performance doesn't work out.
    img_col = patient.img.item()
    seg_col = patient.seg.item()

    img_output = patient.preprocessed_img_path.item()
    seg_output = patient.preprocessed_seg_path.item()   
    target_spacing = (patient.preprocessed_target_spacing_x, patient.preprocessed_target_spacing_y, patient.preprocessed_target_spacing_z)

    if os.path.exists(img_output) and os.path.exists(seg_output):
        print(img_output + " and " + seg_output + " already exists.")
        logger.warning(img_output + " and " + seg_output + " already exists.")
        return patient

    if not os.path.exists(img_col):
        print(img_col + " does not exist.")
        logger.warning(img_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return patient

    if not os.path.exists(seg_col):
        print(seg_col + " does not exist.")
        logger.warning(seg_col + " does not exist.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""
        return patient

    try: 
        img, img_header = load(img_col)
        target_shape = calculate_target_shape(img, img_header, target_spacing)

        img = resample_volume(img, 3, target_shape)
        np.save(img_output, img)
        logger.info("saved img at " + img_output)
        print("saved img at " + img_output)

        seg, _ = load(seg_col)
        seg = interpolate_segmentation_masks(seg, target_shape)
        np.save(seg_output, seg)
        logger.info("saved seg at " + seg_output)
        print("saved seg at " + seg_output)
    except:
        logger.warning("failed to generate resampled volume.")
        print("failed to generate resampled volume.")
        patient['preprocessed_img_path'] = ""
        patient['preprocessed_seg_path'] = ""

    return patient

def interpolate_segmentation_masks(seg, target_shape):
    """
    Use NN interpolation for segmentation masks by resampling boolean masks for each value present.
    :param seg: as numpy.ndarray
    :param target_shape: as tuple or list
    :return: new segmentation as numpy.ndarray
    """
    new_seg = np.zeros(target_shape).astype(int)
    for roi in np.unique(seg):
        if roi == 0:
            continue
        mask = resample_volume(seg == roi, 0, target_shape).astype(bool)
        new_seg[mask] = int(roi)
    return new_seg


def generate_preprocessed_filename(id, suffix, processed_dir, target_spacing_x, target_spacing_y, target_spacing_z):
    """
    Generates target NumPy file path for preprocessed segmentation or acquisition.
    :param idx: case ID
    :param suffix: _seg or _img, depending on which Series is being populated.
    :param processed_dir: path to save .npy files.
    :param target_spacing_x  target x-dimension spacing
    :param target_spacing_y target y-dimension spacing
    :param target_spacing_z target z-dimension spacing
    :return: target file path
    """
    file_name = "".join((processed_dir, str(id), suffix, "_", str(target_spacing_x), "_", str(target_spacing_y), "_", str(target_spacing_z), ".npy"))
    return file_name


def calculate_target_shape(volume, header, target_spacing):
    """
    :param volume: as numpy.ndarray
    :param header: ITK-SNAP header
    :param target_spacing: as tuple or list
    :return: target_shape as list
    """
    src_spacing = header.get_voxel_spacing()
    target_shape = [int(src_d * src_sp / tar_sp) for src_d, src_sp, tar_sp in
                    zip(volume.shape, src_spacing, target_spacing)]
    return target_shape


def resample_volume(volume, order, target_shape):
    """
    Resamples volume using order specified.
    :param volume: as numpy.ndarray
    :param order: 0 for NN (for segmentation), 3 for cubic (recommended for acquisition)
    :param target_shape: as tuple or list
    :return: Resampled volume as numpy.ndarray
    """
    if order == 0:
        anti_alias = False
    else:
        anti_alias = True

    volume = resize(volume, target_shape,
                    order=order, clip=True, mode='edge',
                    preserve_range=True, anti_aliasing=anti_alias)
    return volume


@click.command()
@click.option('-q', '--query', default = None, help = "where clause of SQL query to filter feature table, 'WHERE' does not need to be included, but make sure to wrap with quotes to be interpretted correctly")
@click.option('-b', '--base_directory', type=click.Path(exists=True), required=True)
@click.option('-t', '--target_spacing', nargs=3, type=float, required=True)
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-h', '--hdfs', is_flag=True, default=False, show_default=True, help="(optional) base directory is on hdfs or local filesystem")
@click.option('-n', '--feature_table_output_name', default="feature-table", help= "name of new feature table that is created.")
def cli(spark_master_uri, base_directory, target_spacing, hdfs, query, feature_table_output_name): 

    # Setup Spark context
    import time
    start_time = time.time()
    spark = SparkConfig().spark_session("dl-preprocessing", spark_master_uri, hdfs) 
    generate_feature_table(base_directory, target_spacing, spark, hdfs, query, feature_table_output_name)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_feature_table(base_directory, target_spacing, spark, hdfs, query, feature_table_output_name): 
    annotation_table = os.path.join(base_directory, "tables/annotation")
    scan_table = os.path.join(base_directory, "tables/scan")
    feature_table = os.path.join(base_directory, "features/"+str(feature_table_output_name)+"/")
    feature_dir = os.path.join(base_directory, "features")
    feature_files = os.path.join(base_directory, "features/feature-files/")

    # Load Scan and Annotation tables
    annot_df = spark.read.format("delta").load(annotation_table)
    annot_df.show()
    scan_df = spark.read.format("delta").load(scan_table)
    scan_df.show()


    # Add new columns and save feature tables
    generate_preprocessed_filename_udf = udf(generate_preprocessed_filename, StringType())
    df = annot_df.join(scan_df, ['SeriesInstanceUID'])
    df = df.withColumn("preprocessed_seg_path", lit(generate_preprocessed_filename_udf(df.SeriesInstanceUID, lit('_seg'), lit(feature_files), lit(target_spacing[0]),  lit(target_spacing[1]),  lit(target_spacing[2]) )))
    df = df.withColumn("preprocessed_img_path", lit(generate_preprocessed_filename_udf(df.SeriesInstanceUID, lit('_img'), lit(feature_files), lit(target_spacing[0]),  lit(target_spacing[1]),  lit(target_spacing[2]) )))    
    df = df.withColumn("preprocessed_target_spacing", lit(str(target_spacing)))

    # Add target spacing individually so they can be extracted during row processing
    df = df.withColumn("preprocessed_target_spacing_x", lit(target_spacing[0]))
    df = df.withColumn("preprocessed_target_spacing_y", lit(target_spacing[1]))
    df = df.withColumn("preprocessed_target_spacing_z", lit(target_spacing[2]))
    
    df = df.withColumn("feature_uuid", expr("uuid()"))


    if query:
        sql_query = "SELECT * from feature where " + str(query)
        df.createOrReplaceTempView("feature")  
        try: 
            df = spark.sql(sql_query)
        except Exception as err:
            logger.error("Exception while running spark sql query \"{}\"".format(sql_query), err)
            return
        # If query doesn't return anything, do not proceed.
        if df.count() == 0:
            err_msg = "query \"{}\" has no match. Please revisit your query.".format(query)
            logger.error(err_msg)
            return

    # Resample segmentation and images

    # Create new local directories if needed
    if not os.path.exists(feature_dir) and not hdfs:
        os.mkdir(feature_dir)
    if not os.path.exists(feature_files) and not hdfs:
        os.mkdir(feature_files)
    
    # Preprocess Features Using Pandas DF and applyInPandas() [Apache Arrow]:
    df.groupBy("feature_uuid").applyInPandas(process_patient_pandas_udf, schema = df.schema).show()

    # write table
    df.write.format("delta").mode("overwrite").save(feature_table)

    logger.info("-----Feature table generated:------")
    feature_df = spark.read.format("delta").load(feature_table)
    feature_df.show()

    logger.info("-----Columns Added:------")
    feature_df.select("feature_uuid", "preprocessed_seg_path","preprocessed_img_path", "preprocessed_target_spacing").show(20, False)
    
    logger.info("Feature Table written to ")
    logger.info(feature_table)

if __name__ == "__main__":
    cli()    
