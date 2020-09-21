"""
This module groups dicom images via SeriesInstanceUID, calls a script to generate volumetric images, and interfaces outputs to an IO service.

This module is to be run from the top-level data-processing directory using the -m flag as follows:
Usage:
    $ python3 -m data_processing.process_scan_job --query ... [args]

Parameters:
    ENVIRONMENTAL VARIABLES:
        MIND_ROOT_DIR: The root directory for the delta lake
        MIND_WORK_DIR: POSIX accessable directory for spark workers to use as scratch space
    REQUIRED PARAMETERS:
        --spark_master_uri: spark master uri e.g. spark://master-ip:7077 or local[*]
        --hdfs_uri: HDFS namenode uri e.g. hdfs://master-ip:8020
        --query: a cypher where clause to filter sink IDs based on source ID and relationship fields
        --graph_uri: Neo4j graph URI/bolt Connection
        --tag: Experimental tag for run
        --custom_preprocessing_script: path to preprocessing script to be executed in working directory
    OPTIONAL PARAMETERS:
        All are required.
    EXAMPLE:
        python3 -m data_processing.process_scan_job \
		--query "WHERE source:xnat_accession_number AND source.value='RIA_16-158_001' AND ALL(rel IN r WHERE TYPE(rel) IN ['ID_LINK'])" \
		--spark_master_uri spark://LM620001:7077 \
		--graph_uri bolt://localhost:7687 \
		--hdfs_uri file:// \
		--custom_preprocessing_script /Users/aukermaa/Work/data-processing/data_processing/generateMHD.py \
		--tag test \
"""
import glob, shutil, os, uuid, subprocess, sys, argparse, time

import click
import socket

from paramiko import SSHClient
from scp import SCPClient

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

from pyspark import SQLContext, SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType,StructType,StructField

logger = init_logger()
logger.info("Starting process_scan_job.py")

# pydoop.hdfs.cp("file:///Users/aukermaa/DB/test.txt", "/Users/aukermaa/")

@click.command()
@click.option('-q', '--query', default = None, help = "where clause of CYPHER query to filter ID table, 'WHERE' does not need to be included, wrap with double quotes")
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-g', '--graph_uri', help='spark master uri e.g. bolt://localhost:7883', required=True)
@click.option('-d', '--hdfs_uri', help='hdfs URI uri e.g. hdfs://localhost:8020', required=True)
@click.option('-c', '--custom_preprocessing_script', default = None, help="Path to python file to execute in the working directory")
@click.option('-t', '--tag', default = 'default', help="Provencence tag")
def cli(spark_master_uri, query, hdfs_uri, graph_uri, custom_preprocessing_script, tag):
    """
    This module ....

    Example: python3 ...    """
    # Setup Spark context
    print (query)
    start_time = time.time()
    spark = SparkConfig().spark_session("dicom-to-scan", spark_master_uri)
    generate_scan_table(spark, query, graph_uri, hdfs_uri, custom_preprocessing_script, tag)
    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_scan_table(spark, query, graph_uri, hdfs_uri, custom_preprocessing_script, tag):
    hdfs_db_root    = os.environ["MIND_ROOT_DIR"]
    spark_workspace = os.environ["MIND_WORK_DIR"]
    gpfs_mount      = os.environ['MIND_GPFS_DIR'] 
    print (hdfs_db_root, spark_workspace, gpfs_mount)
    concept_id_TYPE = "SeriesInstanceUID"
    gpfs_host = 'pllimsksparky1'

    # Open a connection to the ID graph database
    logger.info (f'''Conncting to uri={graph_uri}, user="neo4j", pwd="password" ''')
    conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

    sqlc =  SQLContext(spark)

    logger.info ("-------------------------------------- SETUP COMPLETE -------------------------------------------")

    # Begin query/commute process
    df_driver_ids = conn.commute_source_id_to_spark_query(spark, sqlc, WHERE_CLAUSE=query, SINK_TYPE=concept_id_TYPE)
    logger.info (" >>> Graph Query Complete:")
    df_driver_ids.show()

    # Reading dicom and opdata
    df_dcmdata = sqlc.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm_light"))
    df_optdata = sqlc.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm_op_light"))
    df_dcmdata.printSchema()
    df_optdata.printSchema()
    logger.info (" >>> Loaded dicom DB")

    def python_def_generate_mhd(concept_id, input_paths, filenames):
            '''
            Accepts and input of paths, filenames, moves to a temporary working directory, and generates a volumetric MHD filename
            Args:
                    input_paths: list of input paths on HDFS Cluster
                    filenames: list of dcm filenames
            Returns:
                    scan_record_uuid - UUID of the request process
            Notes:
                    Zero error checking, incomplete
            '''


            print ("hello")
            scan_record_uuid  = "SCAN-" + str(uuid.uuid4())
            print (scan_record_uuid)
            # Initialize a working directory
            WORK_DIR   = os.path.join(gpfs_mount + spark_workspace, scan_record_uuid)
            OUTPUT_DIR = os.path.join(WORK_DIR, 'outputs')
            INPUTS_DIR = os.path.join(WORK_DIR, 'inputs')
            print (WORK_DIR)
            os.makedirs(WORK_DIR)
            os.makedirs(OUTPUT_DIR)
            os.makedirs(INPUTS_DIR)

            logger.info(scan_record_uuid)

            # # Data pull request into temporary working directory
            # # TODO:  Will actally be much more streamlined as binary data can be read-out from proxy table
            pull_req_dcm = [ os.path.join(path,file) for path, file in zip(input_paths, filenames)]

            for dcm in pull_req_dcm:
                logger.info(gpfs_mount + dcm, INPUTS_DIR)
                shutil.copy(gpfs_mount + dcm, INPUTS_DIR)
#
#            # Execute some modularized python script
#            # Expects intputs at WORK_DIR, puts outputs into WORK_DIR/outputs
            proc = subprocess.Popen(["/gpfs/mskmindhdp_emc/sw/env/bin/python", custom_preprocessing_script, WORK_DIR], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            print (out, err)

            shutil.rmtree(INPUTS_DIR)
#
#            # Send write message to scan io server
#            # Message format is 5 arguements [command], [search directory path], [concept ID], [record ID], [tag]
#            message = ','.join(["WRITE", OUTPUT_DIR, concept_id, scan_record_uuid, tag])
#            client_socket = socket.socket()  # instantiate
#            client_socket.setblocking(1)
#            client_socket.connect(("pllimsksparky1", 5090))  # connect to the server
#            client_socket.send(message.encode())  # send message
#            client_socket.close()  # close the connection
#
#            # Were all done here, the write service takes care of the rest!!!
#            # Returning record ID
            return scan_record_uuid

    # Make our UDF
    udf_generate_mhd = F.udf(python_def_generate_mhd, StringType())

    # Get ready to run UDF jobs
    df_queue= df_dcmdata.join(df_optdata , ["dicom_record_uuid"]).join(df_driver_ids, [concept_id_TYPE]) \
        .select(concept_id_TYPE,"absolute_hdfs_path","filename") \
        .groupBy(concept_id_TYPE) \
        .agg(F.sort_array( F.collect_list("absolute_hdfs_path")).alias("absolute_hdfs_paths"), \
             F.sort_array( F.collect_list("filename")).alias("filenames") )

    job_start_time = time.time()
    # Run jobs
    logger.info (" >>> Calling jobs on selected patient:")
    df_ct = df_queue.withColumn('payload', udf_generate_mhd(concept_id_TYPE, 'absolute_hdfs_paths', 'filenames'))
    df_ct.show()
    logger.info (" >>> Jobs done")
    logger.info("--- Execute in %s seconds ---" % (time.time() - job_start_time))


if __name__ == "__main__":
    cli()
