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
import os, click, pickle
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../') ))

from sparksession import SparkConfig

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from skimage.transform import resize  
import databricks.koalas as ks      

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit, pandas_udf, PandasUDFType, array
from pyspark.sql.types import StringType


def process_patient_archived(patient,target_spacing):	
    """	
    Given a row with source and destination file paths for a single case, resamples segmentation	
    and acquisition. Also, clips acquisition range to abdominal window.	
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"	
    :return: None	
    """	
    img_col = patient.img	
    seg_col = patient.seg	
    img_output = patient.preprocessed_img_path	
    seg_output = patient.preprocessed_seg_path	

    # target_spacing = patient.preprocessed_target_spacing_x, preprocessed_target_spacing_x,preprocessed_target_spacing_x
    # if patient.preprocessed_img and patient.preprocessed_seg:	
    #     print(img_output + " and " + seg_output + " already exists.")	
    #     return img_output, seg_output	

    img, img_header = load(img_col)	
    target_shape = calculate_target_shape(img, img_header, target_spacing)	

    img = resample_volume(img, 3, target_shape)	
    np.save(img_output, img)	
    print("saved img at " + img_output)	

    seg, _ = load(seg_col)	
    seg = interpolate_segmentation_masks(seg, target_shape)	
    np.save(seg_output, seg)	
    print("saved seg at " + seg_output)	

    return img, seg

def process_patient_ks(df):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """

    img_col = df.img.item()
    # print(img_col)
    seg_col = df.seg.item()
    # img_col = patient.img
    # seg_col = patient.seg
    img_output = df.preprocessed_img_path.item()
    seg_output = df.preprocessed_seg_path.item()

    target_spacing = (df.preprocessed_target_spacing_x, df.preprocessed_target_spacing_y, df.preprocessed_target_spacing_z)

    img, img_header = load(img_col)
    target_shape = calculate_target_shape(img, img_header, target_spacing)
    img = resample_volume(img, 3, target_shape)
    np.save(img_output, img)
    print("saved img at " + img_output)

    seg, _ = load(seg_col)
    seg = interpolate_segmentation_masks(seg, target_shape)
    np.save(seg_output, seg)
    print("saved seg at " + seg_output)

    
    # attempt 1 - directly assign
    # df['preprocessed_img'] = img.tobytes()
    # df['preprocessed_seg'] = seg.tobytes()

    # use assign function
    # preprocessed_img = df.preprocessed_img
    # preprocessed_seg = df.preprocessed_seg
    # df.assign(preprocessed_img = str(img.dumps()))
    # df.assign(preprocessed_seg = str(seg.dumps()))

    return df
    

def process_patient(df):
    """
    Given a row with source and destination file paths for a single case, resamples segmentation
    and acquisition. Also, clips acquisition range to abdominal window.
    :param case_row: pandas DataFrame row with fields "preprocessed_seg_path" and "preprocessed_img_path"
    :return: None
    """
    print("******************IN THE UDF**********************")

    img_col = df.img.item()
    print(img_col)
    seg_col = df.seg.item()
    # img_col = patient.img
    # seg_col = patient.seg

    target_spacing = (df.preprocessed_target_spacing_x, df.preprocessed_target_spacing_y, df.preprocessed_target_spacing_z)

    img, img_header = load(img_col)
    target_shape = calculate_target_shape(img, img_header, target_spacing)
    img = resample_volume(img, 3, target_shape)
    # np.save(img_output, img)
    # print("saved img at " + img_output)

    seg, _ = load(seg_col)
    seg = interpolate_segmentation_masks(seg, target_shape)
    # np.save(seg_output, seg)

    
    # attempt 1 - directly assign
    df['preprocessed_img'] = img.tobytes()
    df['preprocessed_seg'] = seg.tobytes()

    # use assign function
    # preprocessed_img = df.preprocessed_img
    # preprocessed_seg = df.preprocessed_seg
    # df.assign(preprocessed_img = str(img.dumps()))
    # df.assign(preprocessed_seg = str(seg.dumps()))

    return df
    


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


def write_segmented_files(pdf):
    process_patient_ks(pdf)
    return pdf


        
@click.command()
@click.option('-q', '--query', default = None, help = "where clause of SQL query to filter feature table, 'WHERE' does not need to be included, but make sure to wrap with quotes to be interpretted correctly")
@click.option('-b', '--base_directory', type=click.Path(exists=True), required=True)
@click.option('-t', '--target_spacing', nargs=3, type=float, required=True)
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-h', '--hdfs', is_flag=True, default=False, show_default=True, help="(optional) base directory is on hdfs or local filesystem")
@click.option('-n', '--feature_table_output_name', default="feature-table", help= "name of new feature table that is created.")
def cli(spark_master_uri, base_directory, target_spacing, hdfs, query, feature_table_output_name): 
    # Setup Spark context
    spark = SparkConfig().spark_session("dl-preprocessing", spark_master_uri, hdfs)
    generate_feature_table(base_directory, target_spacing, spark, hdfs, query, feature_table_output_name)
    # print("Feature Table written to ", table_dir)

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

    target_spacing_arr = [spacing for spacing in target_spacing]
    df = df.withColumn("preprocessed_seg_path", lit(generate_preprocessed_filename_udf(df.SeriesInstanceUID, lit('_seg'), lit(feature_files), lit(target_spacing[0]),  lit(target_spacing[1]),  lit(target_spacing[2]) )))
    df = df.withColumn("preprocessed_img_path", lit(generate_preprocessed_filename_udf(df.SeriesInstanceUID, lit('_img'), lit(feature_files), lit(target_spacing[0]),  lit(target_spacing[1]),  lit(target_spacing[2]) )))
    # df = df.withColumn("preprocessed_target_spacing", lit(target_spacing_arr))
    df = df.withColumn("preprocessed_target_spacing_x", lit(target_spacing[0]))
    df = df.withColumn("preprocessed_target_spacing_y", lit(target_spacing[1]))
    df = df.withColumn("preprocessed_target_spacing_z", lit(target_spacing[2]))
    if query:
        sql_query = "SELECT * from feature where " + str(query)
        df.createOrReplaceTempView("feature")   
        df = spark.sql(sql_query)

    df.select("preprocessed_seg_path", "preprocessed_img_path", "img", "seg").show()
    # Create new local directories if needed
    if not os.path.exists(feature_dir) and not hdfs:
        os.mkdir(feature_dir)
    if not os.path.exists(feature_files) and not hdfs:
        os.mkdir(feature_files)

    # Call parallel resampling jobs
    # results = Parallel(n_jobs=8)(delayed(process_patient_archived)(row, target_spacing) for row in df.rdd.collect())
    # try with koalas
    ks.set_option("compute.default_index_type", "distributed") 
    kdf = df.to_koalas()  
    kdf = kdf.groupby('annotation_uid').apply(write_segmented_files)


    # Try generating columns before calling pandas UDF
    df = df.withColumn("preprocessed_img",)
    df = df.withColumn("preprocessed_seg",)

    # read the written resampled numpy binaries and merge into delta table
    img_bin_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*_img_*.npy") \
        .option("recursiveFileLookup", "false") \
        .load(feature_files)

    # # show() causes out of memory exception
    # img_bin_df.show(truncate=True)

    img_cond = [df.preprocessed_img_path == img_bin_df.path]
    df = df.join(img_bin_df.select("path", "content"), img_cond)
    df = df.withColumnRenamed("content", "preprocessed_img")


    seg_bin_df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*_seg_*.npy") \
        .option("recursiveFileLookup", "false") \
        .load(feature_files)
    # seg_bin_df.show(truncate=True)
    seg_cond = [df.preprocessed_img_path == seg_bin_df.path]
    df = df.join(seg_bin_df.select("path", "content"), seg_cond)
    df = df.withColumnRenamed("content", "preprocessed_seg")
    df= df.drop('path')


    # play around with using DF's schema or just selecting certain columns to pass into UDF
    # schema = "img string, seg string, preprocessed_img string, preprocessed_seg string"
    # df.groupBy("annotation_uid").applyInPandas(process_patient, schema = (str(df.schema) + 'preprocessed_img binary, preprocessed_seg binary'))
    # df.select("annotation_uid", "preprocessed_img", "preprocessed_seg").show()

    # Write Table to File   
    spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

    # can't write binaries, causes memory errors.
    df.write.format("delta").mode("overwrite").save(feature_table)
    

    print("-----Feature table generated:------")
    feature_df = spark.read.format("delta").load(feature_table)
    feature_df.show()

    print("-----Columns Added:------")
    feature_df.select("preprocessed_img","preprocessed_seg", "preprocessed_target_spacing").show(20, False)
    print("Feature Table written to ", feature_table)

if __name__ == "__main__":
    cli()    

    

