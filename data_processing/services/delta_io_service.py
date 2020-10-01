import os
import subprocess
import argparse
import uuid
import shutil
import glob
import socket
import time
import threading
import os

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

# Parse arguements
parser = argparse.ArgumentParser(description='Start a WRITE SCAN I/O service with a persistent spark context')
parser.add_argument('--hdfs', dest='hdfs', type=str, help='HDFS host (ex. hdfs://pllimsksparky1.mskcc.org:8020)')
parser.add_argument('--host', dest='host', type=str, help='Target host for server (ex. localhost)')
args = parser.parse_args()

hdfs_host       = args.hdfs
io_host         = args.host

hdfs_db_root    = os.environ["MIND_ROOT_DIR"]
spark_workspace = os.environ["MIND_WORK_DIR"]
gpfs_mount      = os.environ["MIND_GPFS_DIR"] 
graph_uri	 = os.environ["GRAPH_URI"]
spark_master_uri = os.environ["SPARK_MASTER_URL"]

# Open a connection to the ID graph database
conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

# Spark setup, persistent spark context for all threads/write/ETL jobs
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.types import StringType,StructType,StructField
import pyarrow as pa

spark = SparkConfig().spark_session("scan-io-service", spark_master_uri)
sqlc = SQLContext(spark)

# Setup logging
logger = init_logger()
logger.info ("Imported spark")


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
        # Annoying
        WORK_DIR    = self.WORK_DIR
        CONCEPT_ID  = self.CONCEPT_ID
        RECORD_ID   = self.RECORD_ID
        TAG_ID      = self.TAG_ID

        CONCEPT_ID_TYPE = 'SeriesInstanceUID'

        # Get output directory to read raw files from
        OUTPUT_DIR = WORK_DIR
        # Set write paths depending on input record type
        if "SCAN" in RECORD_ID:
            RECORD_ID_TYPE = 'scan_record_uuid'
            DATA_PATH = "radiology/scans"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.scans")
        elif "ANNOTATION" in RECORD_ID:
            RECORD_ID_TYPE = 'annotation_record_uuid'
            DATA_PATH = "radiology/annotations"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.annotations")
        elif "FEATURE" in RECORD_ID:
            RECORD_ID_TYPE = 'feature_record_uuid'
            DATA_PATH = "radiology/features"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.features")
        else:
            logger.error("Not a valid or supported record ID " + RECORD_ID)

        # 2. Add files to HDFS/GPFS
        destination_path = os.path.join(hdfs_db_root, DATA_PATH, f"{CONCEPT_ID}/{RECORD_ID}/")
        os.makedirs(gpfs_mount + destination_path)

        # Make sure there is exactly 1 concept ID and 0 record IDs
        if not len(conn.match_concept_node(CONCEPT_ID)) == 1: 
            logger.error("No concept node in DB: " + CONCEPT_ID)
            return
        if not len(conn.match_concept_node(RECORD_ID))  == 0: 
            logger.warning("Identical record already exists: " + RECORD_ID)
            return

        logger.info ("Writing new files:")
        conn.query(f"""MERGE (record_node:{RECORD_ID_TYPE}{{value:'{RECORD_ID}', tag:'{TAG_ID}', status:'PENDING'}})""")

        # Loop over raw files to ingest
        for PAYLOAD_NO, FILE_PATH in enumerate(glob.glob(os.path.join(OUTPUT_DIR,"*"))):
            logger.info("Processing " + FILE_PATH)

            shutil.copy( FILE_PATH , gpfs_mount + destination_path )

            # 3. Prepare an opdata record
            logger.info("Writing record to dt...")
            data_update = {
                RECORD_ID_TYPE: RECORD_ID,
                CONCEPT_ID_TYPE: CONCEPT_ID,
                "payload_number": str(PAYLOAD_NO),
                "absolute_hdfs_path": destination_path,
                "absolute_hdfs_host": hdfs_host,
                "filename": os.path.split(FILE_PATH)[1],
                "type": os.path.splitext(FILE_PATH)[1]
            }

            print ("Appending spark delta table...")
            print (data_update)
            sqlc.createDataFrame([data_update]).write.format("delta").mode("append").option("mergeSchema", "true").save(TABLE_PATH)

        query = f"""
            MATCH (concept_node)
            WHERE concept_node.value = '{CONCEPT_ID}'
            MATCH (record_node:{RECORD_ID_TYPE})
            WHERE record_node.value = '{RECORD_ID}'
            MERGE (concept_node)-[rid:HAS_RECORD]->(record_node)
            SET record_node.status = 'VALID' 
            """
        print (f"\nRunning {query}\n")

        # 5. Integrate in the graph DB
        result = conn.query(query)
        print ("Thread finished.")

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
