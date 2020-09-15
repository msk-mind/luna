"""
This module pre-processes the CE-CT acquisitions and associated segmentations and generates
a DataFrame tracking the file paths of the pre-processed items, stored as NumPy ndarrays.

This module is to be run from the top-level data-processing directory using the -m flag as follows:
Usage: 
    $ python -m data_processing.preprocess_feature --spark_master_uri {spark_master_uri} --base_directory {directory/to/tables} --target_spacing {x_spacing} {y_spacing} {z_spacing}  --query "{sql where clause}" --feature_table_output_name {name-of-table-to-be-created} --custom_preprocessing_script {path/to/preprocessing/script}
    
Parameters: 
    REQUIRED PARAMETERS:
        --target_spacing: target spacing for the x,y,and z dimensions
        --spark_master_uri: spark master uri e.g. spark://master-ip:7077 or local[*]
    OPTIONAL PARAMETERS:
        --base_directory: path to write feature table and files. We assume scan/annotation refined tables are at a specific path on gpfs.
        --query: where clause of SQL query to filter feature tablE. WHERE does not need to be included, make sure to wrap with quotes to be interpretted correctly
            - Queriable Columns to Filter By:
                SeriesInstanceUID
                annotation_record_uuid
                annotation_absolute_hdfs_path
                annotation_filename
                annotation_type
                annotation_payload_number
                annotation_absolute_hdfs_host
                scan_record_uuid
                scan_absolute_hdfs_path
                scan_filename
                scan_type
                scan_absolute_hdfs_host
                scan_payload_number
                preprocessed_annotation_path
                preprocessed_scan_path
                preprocessed_target_spacing_x
                preprocessed_target_spacing_y
                preprocessed_target_spacing_z
                feature_record_uuid
            - examples:
                - filtering by feature_record_uuid: --query "feature_record_uuid='123' or feature_record_uuid='456'"
                - filtering by SeriesInstanceUID: --query "SeriesInstanceUID = '123456abc'"
        --feature_table_output_name: name of feature table that is created, default is feature-table,
                feature table will be created at {base_directory}/tables/features/{feature_table_output_name}
        --custom_preprocessing_script: path to preprocessing script containing "process_patient" function. By default, uses process_patient_default() function for preprocessing
Example:
    $ python -m data_processing.preprocess_feature --spark_master_uri local[*] --base_directory /gpfs/mskmindhdp_emc/user/pateld6/data-processing/test-tables/ --target_spacing 1.0 1.0 3.0  --query "SeriesInstanceUID = '123456abc'" --feature_table_output_name brca-feature-table --custom_preprocessing_script  /gpfs/mskmindhdp_emc/user/pateld6/data-processing/tests/test_external_process_patient_script.py
"""
import os, sys, subprocess, time,importlib
import click
from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

import numpy as np
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from skimage.transform import resize

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, lit, expr
from pyspark.sql.types import StringType

logger = init_logger()
GPFS_MOUNT_DIR = "/gpfs/mskmindhdp_emc"

def generate_absolute_path_from_hdfs(absolute_hdfs_path_col, filename_col):
    # do we need this if?
    if absolute_hdfs_path_col[0] == '/':
        absolute_hdfs_path_col = absolute_hdfs_path_col[1:]
    return os.path.join(GPFS_MOUNT_DIR, absolute_hdfs_path_col, filename_col)

def process_patient_default(patient: pd.DataFrame) -> pd.DataFrame:
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    scan_absolute_hdfs_path = generate_absolute_path_from_hdfs(patient.scan_absolute_hdfs_path.item(), patient.scan_filename.item())
    print("scan path", scan_absolute_hdfs_path)
    annotation_absolute_hdfs_path = generate_absolute_path_from_hdfs(patient.annotation_absolute_hdfs_path.item(), patient.annotation_filename.item())
    print("annot path", annotation_absolute_hdfs_path)
    preprocessed_scan_path = patient.preprocessed_scan_path.item()
    preprocessed_annotation_path = patient.preprocessed_annotation_path.item()   
    target_spacing = (patient.preprocessed_target_spacing_x, patient.preprocessed_target_spacing_y, patient.preprocessed_target_spacing_z)

    if os.path.exists(preprocessed_scan_path) and os.path.exists(preprocessed_annotation_path):
        print(preprocessed_scan_path + " and " + preprocessed_annotation_path + " already exists.")
        logger.warning(preprocessed_scan_path + " and " + preprocessed_annotation_path + " already exists.")
        return patient

    if not os.path.exists(scan_absolute_hdfs_path):
        print(scan_absolute_hdfs_path + " does not exist.")
        logger.warning(scan_absolute_hdfs_path + " does not exist.")
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""
        return patient

    if not os.path.exists(annotation_absolute_hdfs_path):
        print(annotation_absolute_hdfs_path + " does not exist.")
        logger.warning(annotation_absolute_hdfs_path + " does not exist.")
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""
        return patient

    try: 
        img, img_header = load(scan_absolute_hdfs_path)
        target_shape = calculate_target_shape(img, img_header, target_spacing)

        img = resample_volume(img, 3, target_shape)
        np.save(preprocessed_scan_path, img)
        logger.info("saved img at " + preprocessed_scan_path)
        print("saved img at " + preprocessed_scan_path)

        seg, _ = load(annotation_absolute_hdfs_path)
        seg = interpolate_segmentation_masks(seg, target_shape)
        np.save(preprocessed_annotation_path, seg)
        logger.info("saved seg at " + preprocessed_annotation_path)
        print("saved seg at " + preprocessed_annotation_path)
    except Exception as err:
        logger.warning("failed to generate resampled volume.", err)
        print("failed to generate resampled volume.", err)
        patient['preprocessed_scan_path'] = ""
        patient['preprocessed_annotation_path'] = ""

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


def generate_preprocessed_filename(id, suffix, processed_dir):
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
    file_name = "".join((processed_dir, str(id), suffix, ".npy"))
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

def lookup_dmp_patient_id(conn, spark_context, sql_context, SeriesInstanceUID):
    dmp_patient_id = conn.create_id_lookup_table(spark_context, sql_context, "SeriesInstanceUID", "dmp_patient_id", SeriesInstanceUID).collect()
    if dmp_patient_id and len (dmp_patient_id) >= 1:
        return dmp_patient_id[0][1]
    return ""

@click.command()
@click.option('-q', '--query', default = None, help = "where clause of SQL query to filter feature table, 'WHERE' does not need to be included, but make sure to wrap with quotes to be interpretted correctly")
@click.option('-b', '--base_directory', type=click.Path(exists=True), default="/gpfs/mskmindhdp_emc/", help="location to find scan/annotation tables and to create feature table")
@click.option('-t', '--target_spacing', nargs=3, type=float, required=True, help="target spacing for x,y and z dimensions")
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-n', '--feature_table_output_name', default="feature-table", help="name of new feature table that is created.")
@click.option('-c', '--custom_preprocessing_script', default = None, help="Path to python file containing custom 'process_patient' method. By default, uses process_patient_default() for preprocessing")
def cli(spark_master_uri, base_directory, target_spacing, query, feature_table_output_name, custom_preprocessing_script): 
    """
    This module pre-processes the CE-CT acquisitions and associated segmentations and generates
    a DataFrame tracking the file paths of the pre-processed items, stored as NumPy ndarrays.

    Example: python preprocess_feature.py --spark_master_uri {spark_master_uri} --base_directory {directory/to/tables} --target_spacing {x_spacing} {y_spacing} {z_spacing} --query "{sql where clause}" --feature_table_output_name {name-of-table-to-be-created}
    """
    # Setup Spark context
    import time
    start_time = time.time()
    spark = SparkConfig().spark_session("dl-preprocessing", spark_master_uri) 
    generate_feature_table(base_directory, target_spacing, spark, query, feature_table_output_name, custom_preprocessing_script)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_feature_table(base_directory, target_spacing, spark, query, feature_table_output_name, custom_preprocessing_script): 
    annotation_table = os.path.join(base_directory, "data/radiology/tables/radiology.annotations")
    scan_table = os.path.join(base_directory, "data/radiology/tables/radiology.scans")
    feature_table = os.path.join(base_directory, "data/radiology/tables/radiology."+str(feature_table_output_name)+"/")
    feature_files = os.path.join(base_directory, "data/radiology/features/"+str(feature_table_output_name)+"/")
    
    # Load Annotation table and rename columns before merge
    annot_df = spark.read.format("delta").load(annotation_table)
    rename_annotation_columns = ["absolute_hdfs_path", "absolute_hdfs_host", "filename", "type","payload_number"]
    for col in rename_annotation_columns:
        annot_df = annot_df.withColumnRenamed(col,("annotation_"+col))
    annot_df.show(truncate=False)

    # Load Scan Table, filter by mhd [no zraw] and rename columns for merging
    scan_df = spark.read.format("delta").load(scan_table)
    rename_scan_columns = ["absolute_hdfs_path", "absolute_hdfs_host", "filename", "type","payload_number", "item_number"]
    for col in rename_scan_columns:
        scan_df = scan_df.withColumnRenamed(col,("scan_"+col))
    scan_df.createOrReplaceTempView("scan")  
    scan_df = spark.sql("SELECT * from scan where scan_type='.mhd'")
    scan_df.show(truncate=False)


    # join scan and annotation tables 
    generate_preprocessed_filename_udf = udf(generate_preprocessed_filename, StringType())
    df = annot_df.join(scan_df, ['SeriesInstanceUID'])
    df = df.withColumn("preprocessed_annotation_path", lit(generate_preprocessed_filename_udf(df.annotation_record_uuid, lit('_annotation'), lit(feature_files) )))
    df = df.withColumn("preprocessed_scan_path", lit(generate_preprocessed_filename_udf(df.scan_record_uuid, lit('_scan'), lit(feature_files) )))    
    
    # Add target spacing individually so they can be extracted during row processing
    df = df.withColumn("preprocessed_target_spacing_x", lit(target_spacing[0]))
    df = df.withColumn("preprocessed_target_spacing_y", lit(target_spacing[1]))
    df = df.withColumn("preprocessed_target_spacing_z", lit(target_spacing[2]))
    df = df.withColumn("feature_record_uuid", expr("uuid()"))
    df.show(truncate=False)
    # sql processing on joined table if specified
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
    if not os.path.exists(feature_files):
        os.makedirs(feature_files)

    # Preprocess Features Using Pandas DF and applyInPandas() [Apache Arrow]:
    if custom_preprocessing_script:
        # use external preprocessing script

        # add path to python os sys
        sys.path.append(os.path.dirname(custom_preprocessing_script))
        
        # import python file containing process_patient method (without the .py extension)
        custom_preprocessing_script_base = os.path.basename(custom_preprocessing_script.replace(".py", ""))
        module = importlib.import_module(custom_preprocessing_script_base)
        process_patient_func = getattr(module, "process_patient")
        spark.sparkContext.addPyFile(custom_preprocessing_script)

        # call custom preprocessing function
        df = df.groupBy("feature_record_uuid").applyInPandas(process_patient_func, schema = df.schema)
    else:
        # use default preprocessing function  (process_patient_default)
        df = df.groupBy("feature_record_uuid").applyInPandas(process_patient_default, schema = df.schema)

    # Join with clinical proxy tables
    # setup contexts for graph DB
    conn = Neo4jConnection(uri='bolt://dlliskimind1.mskcc.org:7687', user="neo4j", pwd="password")
    sql_context = SQLContext(spark)

    # Add dmp_patient_id column
    uid_join_table = df.select("SeriesInstanceUID")
    uid_join_table = uid_join_table.toPandas()
    uid_join_table["dmp_patient_id"] = uid_join_table.apply(lambda x: lookup_dmp_patient_id(conn, spark.sparkContext, sql_context, x.SeriesInstanceUID), axis=1) 
    uid_join_table = spark.createDataFrame(uid_join_table)
    df = df.join(uid_join_table, ['SeriesInstanceUID'])
    
    # Load Clinical Data, rename table-specific uuid columns, and join tables by dmp_patient_id
    diagnosis_table = os.path.join(base_directory, "data/clinical/tables/clinical.diagnosis")
    diagnosis_df = spark.read.format("delta").load(diagnosis_table)
    diagnosis_df = diagnosis_df.withColumnRenamed("uuid", "diagnosis_uuid")
    df = df.join(diagnosis_df, ['dmp_patient_id'])

    medications_table = os.path.join(base_directory, "data/clinical/tables/clinical.medications")
    medications_df = spark.read.format("delta").load(medications_table)
    medications_df = medications_df.withColumnRenamed("uuid", "medications_uuid")
    df = df.join(medications_df, ['msk_mind_patient_id', 'dmp_patient_id'])

    patients_table = os.path.join(base_directory, "data/clinical/tables/clinical.patients")
    patients_df = spark.read.format("delta").load(patients_table)
    patients_df = patients_df.withColumnRenamed("uuid", "patients_uuid")
    df = df.join(patients_df, ['msk_mind_patient_id', 'dmp_patient_id'])
    df.show()

    # write table
    df.write.format("delta").mode("overwrite").save(feature_table)

    # verify table produced is valid
    logger.info("-----Feature table generated:------")
    feature_df = spark.read.format("delta").load(feature_table)
    feature_df.show()

    logger.info("-----Columns Added:------") 
    feature_df.select("feature_record_uuid", "preprocessed_annotation_path","preprocessed_scan_path", "preprocessed_target_spacing_y","preprocessed_target_spacing_x","preprocessed_target_spacing_z").show(20, False)

    logger.info("Feature Table written to ")
    logger.info(feature_table)

if __name__ == "__main__":
    cli()    
