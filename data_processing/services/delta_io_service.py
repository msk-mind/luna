import os
import subprocess
import argparse
import uuid
import shutil
import glob
import socket
import time
import threading
from pyspark import SQLContext

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger

from checksumdir import dirhash
from filehash import FileHash        


# Parse arguements
# TODO: Use click instead of ArgumentParser
parser = argparse.ArgumentParser(description='Start a WRITE SCAN I/O service with a persistent spark context')
parser.add_argument('--hdfs', dest='hdfs', type=str, help='HDFS host (ex. hdfs://some_ip:8020)')
parser.add_argument('--host', dest='host', type=str, help='Target host for server (ex. localhost)')
args = parser.parse_args()

hdfs_host       = args.hdfs
io_host         = args.host
port = 5090  # initiate port no above 1024

hdfs_db_root    = os.environ["MIND_ROOT_DIR"]
gpfs_mount      = os.environ["MIND_GPFS_DIR"] 
graph_uri	 = os.environ["GRAPH_URI"]
spark_master_uri = os.environ["SPARK_MASTER_URL"]

# Open a connection to the ID graph database
conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

# Spark setup, persistent spark context for all threads/write/ETL jobs
spark = SparkConfig().spark_session("delta-io-service", spark_master_uri)
sqlc = SQLContext(spark)

# Setup logging
logger = init_logger('delta-io-service.log')
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
        logger.info ("New thread")

    def run(self):
        ''' Run in a separate thread '''
        # Annoying
        WORK_DIR       = self.WORK_DIR
        CONCEPT_ID     = self.CONCEPT_ID
        DATA_TYPE = self.RECORD_ID
        TAG_ID         = self.TAG_ID

        CONCEPT_ID_TYPE = 'SeriesInstanceUID'

        # Get output directory to read raw files from
        if os.path.isdir(WORK_DIR):
            OUTPUTS_GLOB = glob.glob(os.path.join(WORK_DIR,"*")) 
            record_hash = dirhash(WORK_DIR, "sha256")
        elif os.path.isfile(WORK_DIR):
            OUTPUTS_GLOB = glob.glob(WORK_DIR) 
            record_hash = FileHash('sha256').hash_file(WORK_DIR)
        else:
            logger.error("Not a valid payload path: " + WORK_DIR)
            return

        # Set write paths depending on input record type
        if DATA_TYPE=="SCAN":
            RECORD_ID_TYPE = 'scan_record_uuid'
            DATA_PATH = "radiology/scans"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.scans")
        elif DATA_TYPE=="ANNOTATION":
            RECORD_ID_TYPE = 'annotation_record_uuid'
            DATA_PATH = "radiology/annotations"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.annotations")
        elif DATA_TYPE=="FEATURE":
            RECORD_ID_TYPE = 'feature_record_uuid'
            DATA_PATH = "radiology/features"
            TABLE_PATH = hdfs_host + os.path.join(hdfs_db_root, "radiology/tables/radiology.features")
        else:
            logger.error("Not a valid or supported data type: " + DATA_TYPE)
            return

        RECORD_ID = f"{DATA_TYPE}-{TAG_ID}-{record_hash}"

        # Make sure there is exactly 1 concept ID and 0 record IDs
        if not len(conn.match_concept_node(CONCEPT_ID)) == 1: 
            logger.error("No concept node in DB: " + CONCEPT_ID)
            return
        if not len(conn.match_concept_node(RECORD_ID))  == 0: 
            logger.warning("Identical record already exists: " + RECORD_ID)
            return

        # 2. Add files to HDFS/GPFS
        destination_path = os.path.join(hdfs_db_root, DATA_PATH, f"{CONCEPT_ID}/{RECORD_ID}/")
        os.makedirs(gpfs_mount + destination_path)

        logger.info ("Writing new files:")
        conn.query(f"""MERGE (record_node:{RECORD_ID_TYPE}{{value:'{RECORD_ID}', tag:'{TAG_ID}', status:'PENDING'}})""")

        # Loop over raw files to ingest
        for PAYLOAD_NO, FILE_PATH in enumerate(OUTPUTS_GLOB):
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

            logger.info ("Appending spark delta table...")
            logger.info ("%s", data_update)
            # Append JSON formatted data_update to delta table
            sqlc.createDataFrame([data_update]).write.format("delta").mode("append").option("mergeSchema", "true").save(TABLE_PATH)

        query = f"""
            MATCH (concept_node)
            WHERE concept_node.value = '{CONCEPT_ID}'
            MATCH (record_node:{RECORD_ID_TYPE})
            WHERE record_node.value = '{RECORD_ID}'
            MERGE (concept_node)-[rid:HAS_RECORD]->(record_node)
            SET record_node.status = 'VALID' 
            """
        logger.info (f"\nRunning {query}\n")

        # 5. Integrate in the graph DB
        result = conn.query(query)
        logger.info ("Thread finished.")

# Main server program
def server_program():
    logger.info (f"delta_io_service is STARTING on {io_host} at {port}")

    server_socket = socket.socket()  # get instance
    server_socket.bind((io_host, port))  # bind host address and port together

    server_socket.setblocking(1)
    server_socket.settimeout(None)

    # configure how many client the server can listen simultaneously
    server_socket.listen(128)
    logger.info (f"delta_io_service is LISTENING on {io_host} at {port}")

    while True:
        conn, address = server_socket.accept()  # accept new connection
        logger.info ("Connection from: " + str(address))
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            break
        action, work_dir, concept_id, record_id, tag_id = data.split(",")
        logger.info("Message from connected user: " + data)

        # Try executing command in a thread
	# This whole driver part will likely change, waiting for  that
        try:
            newthread = ClientThread(work_dir, concept_id, record_id, tag_id)
            newthread.start()
        except:
            logger.error ("Something failed!")

        time.sleep(1)

    conn.close()  # Close the connection
    logger.info (f"delta_io_service is CLOSED on {io_host} at {port}")


if __name__ == "__main__":
    logger.info ("Starting...")
    server_program()
