"""
This module groups dicom images via SeriesInstanceUID, calls a script to generate volumetric images, and interfaces outputs to an IO service.

This module is to be run from the top-level data-processing directory using the -m flag as follows:
Usage:
    $ python3 -m data_processing.process_scan_job \
		--query "WHERE source:xnat_accession_number AND source.value='RIA_16-158_001' AND ALL(rel IN r WHERE TYPE(rel) IN ['ID_LINK'])" \
		--hdfs_uri file:// \
		--custom_preprocessing_script /Users/aukermaa/Work/data-processing/data_processing/generateMHD.py \
		--tag aukerman.test \

Parameters:
    ENVIRONMENTAL VARIABLES:
        MIND_ROOT_DIR: The root directory for the delta lake
        MIND_WORK_DIR: POSIX accessable directory for spark workers to use as scratch space
        MIND_GPFS_DIR: gpfs mount directory on the gpfs cluster
        IO_SERVICE_HOST: host where IO service is running
        IO_SERVICE_PORT: post where IO service is running
    REQUIRED PARAMETERS:
        --hdfs_uri: HDFS namenode uri e.g. hdfs://master-ip:8020
        --query: a cypher where clause to filter sink IDs based on source ID and relationship fields
        --tag: Experimental tag for run
        --custom_preprocessing_script: path to preprocessing script to be executed in working directory
    OPTIONAL PARAMETERS:
        All are required.
"""
import glob, shutil, os, uuid, subprocess, sys, argparse, time

import click
import socket

from checksumdir import dirhash

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

from pyspark import SQLContext, SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType,StructType,StructField

logger = init_logger()
logger.info("Starting process_scan_job.py")

max_retries = 10

# pydoop.hdfs.cp("file:///Users/aukermaa/DB/test.txt", "/Users/aukermaa/")

@click.command()
@click.option('-q', '--query', default = None, help = "where clause of CYPHER query to filter ID table, 'WHERE' does not need to be included, wrap with double quotes")
@click.option('-d', '--hdfs_uri', help='hdfs URI uri e.g. hdfs://localhost:8020', required=True)
@click.option('-c', '--custom_preprocessing_script', default = None, help="Path to python file to execute in the working directory")
@click.option('-t', '--tag', default = 'default', help="Provencence tag")
def cli(query, hdfs_uri,  custom_preprocessing_script, tag):
    """
    This module groups dicom images via SeriesInstanceUID, calls a script to generate volumetric images, and interfaces outputs to an IO service.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Example:
    $ python3 -m data_processing.process_scan_job \
		--query "WHERE source:xnat_accession_number AND source.value='RIA_16-158_001' AND ALL(rel IN r WHERE TYPE(rel) IN ['ID_LINK'])" \
		--hdfs_uri file:// \
		--custom_preprocessing_script /Users/aukermaa/Work/data-processing/data_processing/generateMHD.py \
		--tag aukerman.test \

    """
    # Get environment variables
    spark_master_uri = os.environ["SPARK_MASTER_URL"]

    # Setup Spark context
    print (query)
    start_time = time.time()
    spark = SparkConfig().spark_session("dicom-to-scan", spark_master_uri)
    generate_scan_table(spark, query, hdfs_uri, custom_preprocessing_script, tag)
    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def generate_scan_table(spark, query,  hdfs_uri, custom_preprocessing_script, tag):

    # Get environment variables
    hdfs_db_root     = os.environ["MIND_ROOT_DIR"]
    spark_workspace  = os.environ["MIND_WORK_DIR"]
    gpfs_mount       = os.environ["MIND_GPFS_DIR"] 
    graph_uri    = os.environ["GRAPH_URI"]
    bin_python   = os.environ["PYSPARK_PYTHON"]
    io_service_host = os.environ["IO_SERVICE_HOST"]
    io_service_port = int(os.environ["IO_SERVICE_PORT"])

    concept_id_type = "SeriesInstanceUID"

    # Open a connection to the ID graph database
    logger.info (f'''Conncting to uri={graph_uri}, user="neo4j", pwd="password" ''')
    conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

    sqlc =  SQLContext(spark)

    logger.info ("-------------------------------------- SETUP COMPLETE -------------------------------------------")
    # Begin query/commute process

    df_driver_ids = conn.commute_source_id_to_spark_query(spark, sqlc, WHERE_CLAUSE=query, SINK_TYPE=concept_id_type)
    logger.info (" >>> Graph Query Complete")

    # Reading dicom and opdata
    df_dcmdata = spark.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm"))
    df_optdata = spark.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm_op"))
    logger.info (" >>> Loaded dicom DB")

    def python_def_generate_mhd(concept_id, input_paths, filenames):
            '''
            Accepts and input of paths, filenames, moves to a temporary working directory, and generates a volumetric MHD filename
            Args:
                    input_paths: list of input paths on HDFS Cluster
                    filenames: list of dcm filenames
            Returns:
                    scan_record_uuid - UUID of the request process
            '''


            job_uuid  = "job-" + str(uuid.uuid4())
            print ("Starting " + job_uuid)

            # Initialize a working directory
            WORK_DIR   = os.path.join(gpfs_mount + spark_workspace, job_uuid)
            OUTPUT_DIR = os.path.join(WORK_DIR, 'outputs')
            INPUTS_DIR = os.path.join(WORK_DIR, 'inputs')
            print (f"{job_uuid} - Workdir={WORK_DIR}")
            os.makedirs(WORK_DIR)
            os.makedirs(OUTPUT_DIR)
            os.makedirs(INPUTS_DIR)


            # Data pull request into temporary working directory
            # TODO:  Will actally be much more streamlined as binary data can be read-out from proxy table
            pull_req_dcm = [ gpfs_mount + os.path.join(path,file) for path, file in zip(input_paths, filenames)]

            for dcm in pull_req_dcm:
                print (f"{job_uuid} - Pulling " + dcm)
                shutil.copy(dcm, INPUTS_DIR)

            # Execute some modularized python script
            # Expects inputs at WORK_DIR, puts outputs into WORK_DIR/outputs
            proc = subprocess.Popen([bin_python, custom_preprocessing_script, WORK_DIR], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            print (f"{job_uuid} - Output from script: {out}")
            print (f"{job_uuid} - Errors from script: {err}")

            shutil.rmtree(INPUTS_DIR)
            scan_record_uuid = "-".join(["SCAN", tag, dirhash(OUTPUT_DIR, "sha256")])

            # Send write message to scan io server
            # Message format is 5 arguements [command], [search directory path], [concept ID], [record ID], [tag]
            message = ','.join(["WRITE", OUTPUT_DIR, concept_id, "SCAN", tag])
            print (f"{job_uuid} - Connecting to IO service with {message}")

            # Were all done here, the write service takes care of the rest!!!
            retries = 0
            connected = False
            while not connected and retries < max_retries:
                try:
                    client_socket = socket.socket()  # instantiate
                    client_socket.setblocking(1)
                    client_socket.connect((io_service_host, int(io_service_port)))  # connect to the server
                    client_socket.send(message.encode())  # send message
                    client_socket.close()  # close the connection
                    connected = True
                except:
                    logger.warning(f"{job_uuid} - Could not connect to IO service, trying again in 5 seconds!")
                    time.sleep(5)
                    retries += 1

            if not connected: print (f"{job_uuid} - Abandoned IO service, your results will not be saved") 

            # Returning record ID
            return scan_record_uuid

    # Make our UDF
    spark.sparkContext.addPyFile(custom_preprocessing_script)
    udf_generate_mhd = F.udf(python_def_generate_mhd, StringType())

    # Get ready to run UDF jobs
    df_queue= df_dcmdata.join(df_optdata , ["dicom_record_uuid"]).join(df_driver_ids, [concept_id_type]) \
        .select(concept_id_type,"absolute_hdfs_path","filename") \
        .groupBy(concept_id_type) \
        .agg(F.sort_array( F.collect_list("absolute_hdfs_path")).alias("absolute_hdfs_paths"), \
             F.sort_array( F.collect_list("filename")).alias("filenames") )

    logger.info("Jobs to run: {0}".format(df_queue.count()))

    job_start_time = time.time()
    # Run jobs
    logger.info (" >>> Calling jobs on selected patient:")
    df_ct = df_queue.withColumn('payload', udf_generate_mhd(concept_id_type, 'absolute_hdfs_paths', 'filenames'))
    df_ct.select("SeriesInstanceUID", "payload").show(200, truncate=False)
    logger.info (" >>> Jobs done")
    logger.info("--- Execute in %s seconds ---" % (time.time() - job_start_time))


if __name__ == "__main__":
    cli()
