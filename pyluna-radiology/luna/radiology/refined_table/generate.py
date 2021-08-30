"""
This module generates a volumentric scan image for a given SeriesInstanceUID within a project

Parameters:
    ENVIRONMENTAL VARIABLES:
        MIND_ROOT_DIR: Root directory for *PROJECT* folders 
    REQUIRED PARAMETERS:
        --hdfs_uri: HDFS namenode uri e.g. hdfs://namenode-ip:8020
        --uid: a SeriesInstanceUID
        --tag: Experimental tag for run
        --custom_preprocessing_script: path to preprocessing script
        --project_name: MIND project address
        --file_ext: image file type to generate, mhd or nrrd
        --config_file: application configuration yaml
    OPTIONAL PARAMETERS:
        All are required.
"""
import glob, shutil, os, uuid, subprocess, sys, argparse, time

import click

from checksumdir import dirhash

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.common.custom_logger import init_logger
import luna.common.constants as const

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType,StringType,StructType,StructField

logger = init_logger()
logger.info("Starting process_scan_job.py")
APP_CFG='APP_CFG'



def validate_file_ext(ctx, param, value):
    if not value in ['mhd','nrrd']:
        raise click.UsageError("file_ext should be one of mhd|nrrd")
    else:
        return value.lower()

@click.command()
@click.option('-d', '--hdfs_uri', default='file:///', help='hdfs URI uri e.g. hdfs://localhost:8020', required=True)
@click.option('-c', '--custom_preprocessing_script', help="Path to python file to execute in the working directory", required=True)
@click.option('-t', '--tag', default = 'default', help="Provencence tag")
@click.option('-f', '--config_file', default = 'config.yaml', help="config file")
@click.option('-i', '--uid', help = "SeriesInstanceUID")
@click.option('-p', '--project_name', help="MIND project address")
@click.option('-e', '--file_ext', callback=validate_file_ext, help="file format for scan generation", required=True)
def cli(uid, hdfs_uri, custom_preprocessing_script, tag, config_file, project_name, file_ext):
    """
    This module takes a SeriesInstanceUID, calls a script to generate volumetric images, and updates the scan table.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m luna.radiology.refined_table.generate \
	--hdfs_uri file:// \
	--custom_preprocessing_script luna/radiology/refined_table/dicom_to_scan.py \
	--uid 1.2.840.113619......  \
	--project_name OV_16-.... \
	--file_ext mhd \
	--config_file config.yaml
    """
    start_time = time.time()

    ConfigSet(name=APP_CFG, config_file=config_file)
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='dicom-to-scan')
    generate_scan_table(spark, uid, hdfs_uri, custom_preprocessing_script, tag, project_name, file_ext)

    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_scan_table(spark, uid, hdfs_uri, custom_preprocessing_script, tag, project_name, file_ext):

    # Get environment variables
    hdfs_db_root = os.environ["MIND_ROOT_DIR"]
    bin_python   = os.environ["PYSPARK_PYTHON"]

    concept_id_type = "SeriesInstanceUID"

    project_dir = os.path.join(hdfs_db_root, project_name)
    output_dir = os.path.join(project_dir, const.SCANS)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df_dcmdata = spark.read.format("delta").load( hdfs_uri + os.path.join(project_dir, const.DICOM_TABLE))
    except Exception as ex:
        logger.error("Problem loading dicom table at " + hdfs_uri + os.path.join(project_dir, const.DICOM_TABLE))
        logger.error(ex)
        exit(1)
    logger.info (" >>> Loaded dicom table")

    def python_def_generate_scan(project_dir, file_ext, path):
            '''
            Accepts project path, file type to generate, and dicom path, and generates a volumetric MHD filename
            Args:
                    project_dir: project location
                    file_ext: mhd or nrrd
                    path: path to the dicoms in interest
            Returns:
                    scan_meta: array of (scan_record_uuid, filepath, filetype)
            '''
            scan_meta = []
            job_uuid  = "job-" + str(uuid.uuid4())
            print ("Starting " + job_uuid)

            input_dir, filename  = os.path.split(path)
            input_dir = input_dir[input_dir.index("/"):]
            # Execute some modularized python script
            print ([bin_python, custom_preprocessing_script, project_dir, input_dir, file_ext])
            proc = subprocess.Popen([bin_python, custom_preprocessing_script, project_dir, input_dir, file_ext], \
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()

            print (f"{job_uuid} - Output from script: {out}")

            if proc.returncode != 0:
                print (f"{job_uuid} - Errors from script: {err}")
                return scan_meta

            scan_record_uuid = "-".join(["SCAN", tag, dirhash(input_dir, "sha256")])

            filepath = out.decode('utf-8').split('\n')[-1]

            if file_ext == 'mhd':
                scan_meta = [(scan_record_uuid, filepath+'.mhd', 'mhd'), (scan_record_uuid, filepath+'.zraw', 'zraw')]
            elif file_ext == 'nrrd':
                scan_meta = [(scan_record_uuid, filepath+'.nrrd', 'nrrd')]
            print(scan_meta)
            return scan_meta

    # Make our UDF
    schema = ArrayType(
        StructType([
            StructField('scan_record_uuid', StringType(), False),
            StructField('filepath', StringType(), False),
            StructField('filetype', StringType(), False)
        ]),
    )

    spark.sparkContext.addPyFile(custom_preprocessing_script)
    udf_generate_scan = F.udf(python_def_generate_scan, schema)

    # Filter dicom table with the given SeriesInstanceUID and return 1 row. (Assumption: Dicom folders are organized by SeriesInstanceUID)
    df = df_dcmdata \
        .filter(F.col("metadata."+concept_id_type)==uid) \
        .limit(1)

    if df.count()==0: 
        logger.error("No matching scan for SeriesInstanceUID = " + uid)
        exit(1)

    # Run jobs
    with CodeTimer(logger, 'Generate scans:'):

        df_scan = df.withColumn("scan_data", udf_generate_scan(F.lit(project_dir), F.lit(file_ext), df.path))
        # expand the array and flatten the schema
        df_scan = df_scan.withColumn("exp", F.explode("scan_data"))
        df_scan = df_scan.select("metadata.SeriesInstanceUID", "exp.*")

        # Check if the same scan_record_uuid/filetype combo exists. if not, append to scan table.
        scan_table_path = os.path.join(project_dir, const.SCAN_TABLE)

        if os.path.exists(scan_table_path):
            df_existing_scan = spark.read.format("delta").load(scan_table_path)
            intersection = df_existing_scan.join(F.broadcast(df_scan), ["scan_record_uuid", "filetype"])

            if intersection.count() == 0:
                df_scan.write.format("delta") \
                    .mode("append") \
                    .save(scan_table_path)
        else:
            df_scan.write.format("delta") \
                    .mode("append") \
                    .save(scan_table_path)

    # Validation step
    df_scan.show(200, truncate=False)


if __name__ == "__main__":
    cli()
