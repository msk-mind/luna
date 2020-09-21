import os
import subprocess
import logging
import argparse
import uuid
import shutil
from data_processing.common.Neo4jConnection import Neo4jConnection
import logging
import glob
import socket
import time
import logging
import threading
import os
import datetime
import time

# Parse arguements
parser = argparse.ArgumentParser(description='Start a WRITE SCAN I/O service with a persistent spark context')
parser.add_argument('--hdfs', dest='hdfs', type=str, help='HDFS host (ex. hdfs://pllimsksparky1.mskcc.org:8020)')
parser.add_argument('--spark', dest='spark', type=str, help='Spark Cluster URI to use (ex. spark://localhost:7070)')
parser.add_argument('--graph', dest='graph', type=str, help='Graph DB URI (ex. bolt://dlliskimind1.mskcc.org:7687)')
parser.add_argument('--host', dest='host', type=str, help='Target host for server (ex. localhost)')
args = parser.parse_args()

hdfs_host       = args.hdfs
spark_cluster   = args.spark
graph_host      = args.graph
io_host         = args.host


hdfs_db_root    = os.environ["MIND_ROOT_DIR"]

# Open a connection to the ID graph database
conn = Neo4jConnection(uri=graph_host, user="neo4j", pwd="password")

# Setup spark context and processing
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="/usr/local/bin/python3"

# Spark setup, persistent spark context for all threads/write/ETL jobs
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.types import StringType,StructType,StructField
import pyarrow as pa
conf = SparkConf()\
    .setAppName("scan_io_service") \
    .setMaster(spark_cluster) \
    .set("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
    .set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .set("spark.driver.bindAddress", "0.0.0.0") \
    .set("spark.driver.memory", "2g") \
    .set("spark.executor.memory", "2g") \
    .set("spark.cores.max", "1")

sc = SparkContext(conf=conf)
sqlc =  SQLContext(sc)
#
# sc = SparkSession.builder \
#     .appName('scan_io_service') \
#     .config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
#     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
#     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
#     .config("spark.driver.bindAddress", "0.0.0.0") \
#     .config("spark.driver.memory", "4g") \
#     .config("spark.cores.max", "6") \
#     .getOrCreate()
#
# print ("SPARK VERSION ", sc.version)
# sqlc =  SQLContext(sc)

# Setup logging
logging.basicConfig(filename='/Users/aukermaa/DB/logs/io_service_log.txt',level=logging.INFO)
logging.info ("Imported spark")

#
# from delta.tables import DeltaTable
# scan_table_dthandle     = DeltaTable.forPath(sc, hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.scans"))
#

def getBlock():
    c_timestamp = datetime.datetime.now()
    return (c_timestamp.year - 2000)*365*60*24 + c_timestamp.day*60*24 + c_timestamp.hour*60 + c_timestamp.minute

class ClientThread(threading.Thread):
    ''' Basic thread class '''
    def __init__(self, WORK_DIR, CONCEPT_ID, RECORD_ID, TAG_ID):
        '''
        Initalize the thread class with arguements
        args:
            WORK_DIR: working directory path where raw files exist at /outputs
            CONCEPT_ID: All records should be attached to a concept ID, e.g. a scan series id
            RECORD_ID: All records should have a pre-assigned record ID
            TAG_ID: All new records get a tag
        '''
        threading.Thread.__init__(self)
        self.WORK_DIR   = WORK_DIR
        self.CONCEPT_ID = CONCEPT_ID
        self.RECORD_ID  = RECORD_ID
        self.TAG_ID     = TAG_ID
        print ("New thread")

    def run(self):
        ''' Run in a separate thread '''
        start_time = time.time()
        # Annoying
        WORK_DIR    = self.WORK_DIR
        CONCEPT_ID  = self.CONCEPT_ID
        RECORD_ID   = self.RECORD_ID
        TAG_ID      = self.TAG_ID


        # Get output directory to read raw files from
        OUTPUT_DIR = WORK_DIR

        logging.info ("Files to ingest")
        logging.info (str(glob.glob(os.path.join(OUTPUT_DIR,"*"))))

        # Loop over raw files to ingest
    #    for PAYLOAD_NO, FILE_PATH in enumerate(glob.glob(os.path.join(OUTPUT_DIR,"*"))):

        # Make sure there is an associated concept ID in the graph database
        result = conn.match_concept_node(CONCEPT_ID)
        logging.info(result)

        CONCEPT_ID_TYPE = 'SeriesInstanceUID'

        # Set write paths depending on input record type
        if "SCAN" in RECORD_ID:
            RECORD_ID_TYPE = 'scan_record_uuid'
            DATA_PATH = "radiology/scans"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.scans.object")
        elif "ANNOTATION" in RECORD_ID:
            RECORD_ID_TYPE = 'annotation_record_uuid'
            DATA_PATH = "radiology/annotations"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.annotations.object")
        elif "FEATURE" in RECORD_ID:
            RECORD_ID_TYPE = 'feature_record_uuid'
            DATA_PATH = "radiology/features"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.features.object")
        else:
            logging.error("Not a valid or supported record ID " + RECORD_ID)

        # Main write operations
        if (len(result)) == 1:
            logging.info ("Found concept node! ")

            # 1. Add a new record node to the graph database if it doesn't already exist
            result = conn.query(f"""
                MERGE (record_node:{RECORD_ID_TYPE}{{value:'{RECORD_ID}'}})
                MERGE (tag_node:tag_id{{value:'{TAG_ID}'}})
                """
            )

            address = RECORD_ID

            df1  = sqlc.read.format("binaryFile").option("pathGlobFilter", "*").load(OUTPUT_DIR)

            df2 = sqlc.createDataFrame([{
                        "block": getBlock(),
                        "type": "CT",
                        "SeriesInstanceUID": "1.840.0000001",
                        "dimensions": 3
                    }])


            record = df1.select( F.struct("*").alias("payload")).withColumn("address", F.lit(address)).join(df2.withColumn("address", F.lit(address)), ['address'])
            record.printSchema()

            print ("Writing object record...")
            record.write.format("parquet").mode("append").partitionBy("block").save(TABLE_PATH)

            df1.unpersist()
            df2.unpersist()
            record.unpersist()

            query = f"""
                MATCH (tag_node)
                WHERE tag_node.value = '{TAG_ID}'
                MATCH (concept_node)
                WHERE concept_node.value = '{CONCEPT_ID}'
                MATCH (record_node:{RECORD_ID_TYPE})
                WHERE record_node.value = '{RECORD_ID}'
                MERGE (concept_node)-[rid:HAS_RECORD]->(record_node)
                MERGE (tag_node)-[rtag:TAGGED]->(record_node)
                """
            print (f"\nRunning {query}\n")

            # 5. Integrate in the graph DB
            result = conn.query(query)
        print("--- Thread finished successfully in %s seconds ---" % (time.time() - start_time))

# Main server program
def server_program():
    # get the hostname
    host = io_host
    port = 5090  # initiate port no above 1024
    print (f"io_service is STARTING on {host} at {port}")

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    server_socket.setblocking(1)
    server_socket.settimeout(None)

    # configure how many client the server can listen simultaneously
    server_socket.listen(128)
    print (f"io_service is LISTENING on {host} at {port}")

    while True:
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        args = data.split(",")
        if not data:
            break
        print("Message from connected user: ", args)

        # Try executing command in a thread
        try:
            newthread = ClientThread(args[1], args[2], args[3], args[4])
            newthread.start()
        except:
            print ("Something failed!")

        time.sleep(1)

    conn.close()  # Close the connection


if __name__ == "__main__":
    print ("Starting...")
    server_program()
