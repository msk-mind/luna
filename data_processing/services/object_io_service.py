""" Experimental service """
# TODO: Use click (later)
# TODO: Mirgration and cleanup (later)
import os
import subprocess
import argparse
import uuid
import shutil
import glob
import socket
import time
import threading
import datetime
from pyspark import SQLContext

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

# Parse arguements
parser = argparse.ArgumentParser(description='Start a WRITE SCAN I/O service with a persistent spark context')
parser.add_argument('--hdfs', dest='hdfs', type=str, help='HDFS host (ex. hdfs://localhost:8020)')
parser.add_argument('--spark', dest='spark', type=str, help='Spark Cluster URI to use (ex. spark://localhost:7070)')
parser.add_argument('--graph', dest='graph', type=str, help='Graph DB URI (ex. bolt://localhost:7687)')
parser.add_argument('--host', dest='host', type=str, help='Target host for server (ex. localhost)')
args = parser.parse_args()

hdfs_host       = args.hdfs
spark_uri       = args.spark
graph_host      = args.graph
io_host         = args.host
port = 5090  # initiate port no above 1024

hdfs_db_root    = os.environ["MIND_ROOT_DIR"]

# Open a connection to the ID graph database
conn = Neo4jConnection(uri=graph_host, user="neo4j", pwd="password")

# Spark setup, persistent spark context for all threads/write/ETL jobs
spark = SparkConfig().spark_session("object-io-service", spark_uri)
sqlc = SQLContext(spark)

# Setup logging
logger = init_logger("object-io-service.log")


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

        logger.info ("Files to ingest")
        logger.info (str(glob.glob(os.path.join(OUTPUT_DIR,"*"))))

        # Loop over raw files to ingest
    #    for PAYLOAD_NO, FILE_PATH in enumerate(glob.glob(os.path.join(OUTPUT_DIR,"*"))):

        # Make sure there is an associated concept ID in the graph database
        result = conn.match_concept_node(CONCEPT_ID)
        logger.info(result)

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
            logger.error("Not a valid or supported record ID " + RECORD_ID)

        # Main write operations
        if (len(result)) == 1:
            logger.info ("Found concept node! ")

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
            logger.info(f"\nRunning {query}\n")

            # 5. Integrate in the graph DB
            result = conn.query(query)
        logger.info("--- Thread finished successfully in %s seconds ---" % (time.time() - start_time))

# Main server program
def server_program():
    logger.info(f"io_service is STARTING on {io_host} at {port}")

    server_socket = socket.socket()  # get instance
    server_socket.bind((io_host, port))  # bind host address and port together

    server_socket.setblocking(1)
    server_socket.settimeout(None)

    # configure how many client the server can listen simultaneously
    server_socket.listen(128)
    logger.info(f"io_service is LISTENING on {io_host} at {port}")

    while True:
        conn, address = server_socket.accept()  # accept new connection
        logger.info("Connection from: " + str(address))
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            break
        action, work_dir, concept_id, record_id, tag_id = data.split(",")
        logger.info("Message from connected user: " + data)

        # Try executing command in a thread
        try:
            # WORK_DIR, CONCEPT_ID, RECORD_ID, TAG_ID
            newthread = ClientThread(work_dir, concept_id, record_id, tag_id)
            newthread.start()
        except Exception as ex:
            logger.error("Something failed!: ", ex)

        time.sleep(1)

    conn.close()  # Close the connection


if __name__ == "__main__":
    logger.info("Starting...")
    server_program()
