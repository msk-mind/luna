"""
This module reads the PatientName and SeriesInstanceUID columns of the dcm proxy table and updates the graph database with the proper ID LINK relationship 

It's an additive-idempotematic such that rerunning this script multiple times on the same proxy table produces the same results, but on an updated proxy table differentially updates the graph.

This module is to be run from the top-level data-processing directory using the -m flag as follows:
Usage:
    $ python3 -m data_processing.proxy_to_graph ... [args]

Parameters:
    ENVIRONMENTAL VARIABLES:
        MIND_ROOT_DIR: The root directory for the delta lake
    REQUIRED PARAMETERS:
        --spark_master_uri: spark master uri e.g. spark://master-ip:7077 or local[*]
        --hdfs_uri: HDFS namenode uri e.g. hdfs://master-ip:8020
        --query: a cypher where clause to filter sink IDs based on source ID and relationship fields
    OPTIONAL PARAMETERS:
        All are required.
    EXAMPLE:
        python3 -m data_processing.proxy_to_graph \
		--spark_master_uri spark://LM620001:7077 \
		--graph_uri bolt://localhost:7687 \
		--hdfs_uri file:// 
"""
import glob, shutil, os, uuid, subprocess, sys, argparse, time

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

@click.command()
@click.option('-s', '--spark_master_uri', help='spark master uri e.g. spark://master-ip:7077 or local[*]', required=True)
@click.option('-g', '--graph_uri', help='spark master uri e.g. bolt://localhost:7883', required=True)
@click.option('-h', '--hdfs_uri', help='hdfs URI uri e.g. hdfs://localhost:8020', required=True)
def cli(spark_master_uri, hdfs_uri, graph_uri) :
    """
    This module groups dicom images via SeriesInstanceUID, calls a script to generate volumetric images, and interfaces outputs to an IO service.
    
    This module is to be run from the top-level data-processing directory using the -m flag as follows:

    Usage: python3 -m data_processing.proxy_to_graph \
        --spark_master_uri spark://spark-ip:7077 \
        --graph_uri bolt://localhost:7687 \
        --hdfs_uri file:// 
    """
    # Setup Spark context
    start_time = time.time()
    spark = SparkConfig().spark_session("dicom-to-graph", spark_master_uri)
    update_graph_with_scans(spark, graph_uri, hdfs_uri) 
    logger.info("--- Finished in %s seconds ---" % (time.time() - start_time))


def update_graph_with_scans(spark, graph_uri, hdfs_uri):
    hdfs_db_root    = os.environ["MIND_ROOT_DIR"]
    logger.info (f"hdfs_db_root={hdfs_db_root}")

    # Open a connection to the ID graph database
    logger.info (f'''Conncting to uri={graph_uri}, user="neo4j", pwd="password" ''')
    conn = Neo4jConnection(uri=graph_uri, user="neo4j", pwd="password")

    logger.info ("-------------------------------------- SETUP COMPLETE -------------------------------------------")
    job_start_time = time.time()

    # Reading dicom and opdata
    df_dcmdata = spark.read.format("delta").load( hdfs_uri + os.path.join(hdfs_db_root, "radiology/tables/radiology.dcm"))
    logger.info (" >>> Loaded dicom DB")

    tuple_to_add = df_dcmdata.select("PatientName", "SeriesInstanceUID")\
	.groupBy("PatientName", "SeriesInstanceUID")\
	.count()\
	.withColumnRenamed("PatientName", "xnat_patient_id")\
	.toPandas()

    id_1 = "xnat_patient_id"
    id_2 = "SeriesInstanceUID"
    link = "HAS_SCAN"

    for index, row in tuple_to_add.iterrows():
        query ='''MERGE (id1:{0} {{value: "{1}"}}) MERGE (id2:{2} {{value: "{3}"}}) MERGE (id1)-[r:{4}]->(id2)'''.format(id_1, row[id_1], id_2, row[id_2], link)
        logger.info (query)
        conn.query(query, db="neo4j")
    logger.info (" >>> Jobs done")
    logger.info("--- Execute in %s seconds ---" % (time.time() - job_start_time))


if __name__ == "__main__":
    cli()
