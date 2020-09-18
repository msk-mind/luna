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

import pydoop
import pydoop.hdfs as hdfs
import socket

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.custom_logger import init_logger

# pydoop.hdfs.cp("file:///Users/aukermaa/DB/test.txt", "/Users/aukermaa/")

@click.command()
@click.option('-q', '--query', default = None, help = "where clause of SQL query to filter feature table, 'WHERE' does not need to be included, but make sure to wrap with quotes to be interpretted correctly")
@click.option('-b', '--base_directory', type=click.Path(exists=True), default="/gpfs/mskmindhdp_emc/", help="location to find scan/annotation tables and to create feature table")
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-n', '--feature_table_output_name', default="feature-table", help="name of new feature table that is created.")
@click.option('-c', '--custom_preprocessing_script', default = None, help="Path to python file containing custom 'process_patient' method. By default, uses process_patient_default() for preprocessing")
def cli(spark_master_uri, base_directory, target_spacing, query, feature_table_output_name, custom_preprocessing_script):
    """
    This module ....

    Example: python3 ...    """
    # Setup Spark context
    import time
    start_time = time.time()
    spark = SparkConfig().spark_session("dicom-to-scan", spark_master_uri)
    generate_feature_table(base_directory, target_spacing, spark, query, feature_table_output_name, custom_preprocessing_script)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))




parser = argparse.ArgumentParser(description='Submission script to run dicom-to-scan concept transformations')
parser.add_argument('--id', dest='id', type=str, help='Specify query ID (ex. DMP-0001)')
parser.add_argument('--type', dest='type', type=str, help='Specify query ID type (ex. dmp_patient_id')
parser.add_argument('--py', dest='py', type=str, help='Python script to run (ex. /path/to/generateMHD.py)')
parser.add_argument('--spark', dest='spark', type=str, help='Spark Cluster to use (ex. spark://localhost:7070)')
parser.add_argument('--hdfs', dest='hdfs', type=str, help='HDFS host')
parser.add_argument('--db', dest='db', type=str, help='DB root directory (ex. /DB/')
parser.add_argument('--graph', dest='graph', type=str, help='Graph DB hostname')
parser.add_argument('--gpfs', dest='gpfs', type=str, help='GPFS host')
parser.add_argument('--mount', dest='mount', type=str, help='GPFS mount')
parser.add_argument('--tag', dest='tag', type=str, help='Provencence Tag')
args = parser.parse_args()

print (os.getlogin())

query_type = args.type
query_id   = args.id
pyjob_path   = args.py
spark_cluster   = args.spark
hdfs_host  = args.hdfs
hdfs_db_root = args.db
graph_host      = args.graph
gpfs_mount = args.mount
gpfs_host = args.gpfs
tag_user  = args.tag

# subprocess to find the java , scala and python version
cmd1 = "java -version"
cmd2 = "scala -version"
cmd3 = "python3 --version"

arr = [cmd1, cmd2, cmd3]

for cmd in arr:
    print (cmd)
    subprocess.run (cmd.split(" "))

print("JAVA_HOME=", os.getenv("JAVA_HOME"))
print("HADOOP_HOME=", os.getenv("HADOOP_HOME"))
print("SPARK_HOME=", os.getenv("SPARK_HOME"))
print("HOME=",      os.getenv("HOME"))

# Setup spark context and processing
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/local/bin/python3"

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType,StructType,StructField
import pyarrow as pa
conf = SparkConf()\
    .setAppName("generate_ct") \
    .setMaster(spark_cluster) \
    .set("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
    .set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .set("spark.driver.bindAddress", "0.0.0.0") \
    .set("spark.driver.memory", "4g") \
    .set("spark.cores.max", "8") \
    .set("spark.network.timeout", "600s")

sc = SparkContext(conf=conf)
sqlc =  SQLContext(sc)
print ("SPARK VERSION ", sc.version)

print ("-------------------------------------- SETUP COMPLETE -------------------------------------------")

# We identify scan concepts with the series instance uid
concept_id_TYPE = "SeriesInstanceUID"
SPARK_SCRATCH_SPACE = "/Users/aukermaa/DB/working/"

# Open a connection to the ID graph database
print (f'''Conncting to uri={graph_host}, user="neo4j", pwd="password" ''')
conn = Neo4jConnection(uri=graph_host, user="neo4j", pwd="password")

# Begin query/commute process
print (f" >>> Looking for ID {query_id} of type [{query_type}]")
if query_type == "cohort_id":
    # This is a cohort ID, so only transverse cohort link once
    df_driver_ids = conn.commute_cohort_id_to_spark(sc, sqlc, query_type, concept_id_TYPE, query_id )
elif 'record' in query_type:
    df_driver_ids = conn.commute_record_id_to_spark(sc, sqlc, query_type, concept_id_TYPE)
elif query_type == concept_id_TYPE:
    # The ID given is equal to the concept ID, so no query needed
    if query_id == "all":
        df_driver_ids = conn.commute_all_sink_id(sc, sqlc, concept_id_TYPE)
    else:
        df_driver_ids = conn.commute_sink_id_to_spark(sc, sqlc, concept_id_TYPE, query_id)

else:
    # COmmute id
    df_driver_ids = conn.commute_source_id_to_spark(sc, sqlc, query_type, concept_id_TYPE, query_id)

print (" >>> Graph Query Complete:")
df_driver_ids.show()

# Reading dicom and opdata
df_dcmdata = sqlc.read.format("delta").load( hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm"))
df_optdata = sqlc.read.format("delta").load( hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm_op"))
print (" >>> Loaded dicom DB")
df_dcmdata.printSchema()
df_optdata.printSchema()

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

        import glob
        import shutil
        import os
        import uuid
        import subprocess
        import sys
        from paramiko import SSHClient
        from scp import SCPClient

        sys.path.insert(0, '/usr/local/Cellar/hadoop/3.2.1_1/bin/')

        scan_record_uuid  = "SCAN-" + str(uuid.uuid4())

        # Initialize a working directory
        WORK_DIR   = os.path.join(SPARK_SCRATCH_SPACE, scan_record_uuid)
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
            # Awful and slow bc it spins up a new jvm 100 times per scan being processed
            # subprocess.call(['/bin/sh', '/usr/local/bin/hadoop', 'fs', '-get', hdfs_host + dcm, WORK_DIR])
            # pydoop.hdfs.cp( os.path.join(hdfs_host, dcm), WORK_DIR )

        # Execute some modularized python script
        # Expects intputs at WORK_DIR, puts outputs into WORK_DIR/outputs
        proc = subprocess.Popen(["/usr/local/bin/python3", pyjob_path, WORK_DIR], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        # Send write message to scan io server
        # Message format is 5 arguements [command], [working directory path], [concept ID to attach], [record ID to ingest], [tag]
        message = ','.join(["WRITE", OUTPUT_DIR, concept_id, scan_record_uuid, tag_user])
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

print (" >>> Job queue: ", df_get.count())
df_get.show()

# Run jobs
print (" >>> Calling jobs on selected patient:")
df_ct = df_get.withColumn('payload', udf_generate_mhd(concept_id_TYPE, 'absolute_hdfs_paths', 'filenames'))
print (" >>> Jobs done: ", df_ct.count())
df_ct.show()


if __name__ == "__main__":
    cli()
