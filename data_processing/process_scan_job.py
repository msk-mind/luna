"""
This module groups dicom images via SeriesInstanceUID, calls a script to generate volumetric images, and interfaces outputs to an IO service.

This module is to be run from the top-level data-processing directory using the -m flag as follows ?
Usage:

Parameters:
    REQUIRED PARAMETERS:
        --spark_master_uri: spark master uri e.g. spark://master-ip:7077 or local[*]
    OPTIONAL PARAMETERS:

    EXAMPLE:
"""
import sys, os, subprocess, argparse

import click
import socket

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
# TODO: query?
@click.option('-q', '--query', default = None, help = "where clause of CYPHER query to filter ID table, 'WHERE' does not need to be included, wrap with double quotes")
@click.option('-b', '--base_directory', type=click.Path(exists=True), default="/gpfs/mskmindhdp_emc/", help="location to find scan/annotation tables and to create feature table")
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-g', '--graph_uri', help='spark master uri e.g. bolt://localhost:7883', required=True)
@click.option('-d', '--hdfs_uri', help='hdfs URI uri e.g. hdfs://localhost:8020', required=True)
@click.option('-c', '--custom_preprocessing_script', default = None, help="Path to python file to execute in the working directory")
@click.option('-t', '--tag', default = 'default', help="Provencence tag")
def cli(spark_master_uri, base_directory, query, hdfs_uri, graph_uri, custom_preprocessing_script, tag):
    """
    This module ....

    Example: python3 ...    """
    # Setup Spark context
    print (query)
    import time
    start_time = time.time()
    spark = SparkConfig().spark_session("dicom-to-scan", spark_master_uri)
    generate_scan_table(base_directory, spark, query, graph_uri, hdfs_uri, custom_preprocessing_script, tag)
    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_scan_table(base_directory, spark, query, graph_uri, hdfs_uri, custom_preprocessing_script, tag):
    os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
    os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/local/bin/python3"
    hdfs_db_root    = os.environ["MIND_ROOT_DIR"]
    spark_workspace = os.environ["MIND_WORK_DIR"]

    sc = spark
    sqlc =  SQLContext(spark)

    logger.info ("-------------------------------------- SETUP COMPLETE -------------------------------------------")

    # We identify scan concepts with the series instance uid
    concept_id_TYPE = "SeriesInstanceUID"

    # Open a connection to the ID graph database
    logger.info (f'''Conncting to uri={graph_uri}, user="neo4j", pwd="password" ''')
    conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

    # Begin query/commute process
    # logger.info (f" >>> Looking for ID {query_id} of type [{query_type}]")
    df_driver_ids = conn.commute_source_id_to_spark_query(spark, sqlc, WHERE_CLAUSE=query, SINK_TYPE=concept_id_TYPE)

    logger.info (" >>> Graph Query Complete:")
    df_driver_ids.show()


    gpfs_host = 'localhost'
    gpfs_mount = '/'
    # Reading dicom and opdata
    df_dcmdata = sqlc.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm"))
    df_optdata = sqlc.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm_op"))
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

            import glob, shutil, os, uuid, subprocess, sys
            from paramiko import SSHClient
            from scp import SCPClient

            sys.path.insert(0, '/usr/local/Cellar/hadoop/3.2.1_1/bin/')

            scan_record_uuid  = "SCAN-" + str(uuid.uuid4())

            # Initialize a working directory
            WORK_DIR   = os.path.join(spark_workspace, scan_record_uuid)
            OUTPUT_DIR = os.path.join(WORK_DIR, 'outputs')
            INPUTS_DIR = os.path.join(WORK_DIR, 'inputs')
            os.makedirs(WORK_DIR)
            os.makedirs(OUTPUT_DIR)
            os.makedirs(INPUTS_DIR)

            # # Data pull request into temporary working directory
            # # TODO:  Will actally be much more streamlined as binary data can be read-out from proxy table
            pull_req_dcm = [ os.path.join(path,file) for path, file in zip(input_paths, filenames)]

            #for dcm in pull_req_dcm: shutil.copy(dcm, WORK_DIR)
            ssh = SSHClient()
            ssh.load_system_host_keys()
            ssh.connect(gpfs_host)
            with SCPClient(ssh.get_transport()) as scp:
                for dcm in pull_req_dcm:
                    scp.get(gpfs_mount + dcm, INPUTS_DIR)

            # Execute some modularized python script
            # Expects intputs at WORK_DIR, puts outputs into WORK_DIR/outputs
            proc = subprocess.Popen(["/usr/local/bin/python3", custom_preprocessing_script, WORK_DIR], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()

            # Send write message to scan io server
            # Message format is 5 arguements [command], [search directory path], [concept ID], [record ID], [tag]
            message = ','.join(["WRITE", OUTPUT_DIR, concept_id, scan_record_uuid, tag])
            client_socket = socket.socket()  # instantiate
            client_socket.setblocking(1)
            client_socket.connect(("localhost", 5090))  # connect to the server
            client_socket.send(message.encode())  # send message
            client_socket.close()  # close the connection

            # Were all done here, the write service takes care of the rest!!!
            # Returning record ID
            return scan_record_uuid

    # Make our UDF
    udf_generate_mhd = F.udf(python_def_generate_mhd, StringType())

    # Get ready to run UDF jobs
    df_get = df_dcmdata.join(df_optdata , ["dicom_record_uuid"]).join(df_driver_ids, [concept_id_TYPE]) \
        .select(concept_id_TYPE,"absolute_hdfs_path","filename") \
        .groupBy(concept_id_TYPE) \
        .agg(F.sort_array( F.collect_list("absolute_hdfs_path")).alias("absolute_hdfs_paths"), \
             F.sort_array( F.collect_list("filename")).alias("filenames") )

    logger.info (" >>> Job queue:")
    df_get.show()

    # Run jobs
    logger.info (" >>> Calling jobs on selected patient:")
    df_ct = df_get.withColumn('payload', udf_generate_mhd(concept_id_TYPE, 'absolute_hdfs_paths', 'filenames'))
    logger.info (" >>> Jobs done")
    df_ct.show()


if __name__ == "__main__":
    cli()
