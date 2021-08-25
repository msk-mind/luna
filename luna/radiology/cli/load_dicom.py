'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. load DicomSeries object given a delta table path 

'''

# General imports
import os, json, logging
import click
from pathlib import Path

# From common
from luna.common.custom_logger   import init_logger
from luna.common.DataStore       import DataStore
from luna.common.Node            import Node
from luna.common.config          import ConfigSet

# From common
from luna.common.sparksession     import SparkConfig

cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    load_dicom_with_container(cohort_id, datastore_id, method_data)

def load_dicom_with_container(cohort_id, container_id, method_data):
    """
    Using the container API interface, fill scan with original dicoms from table
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    # Do some setup
    datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
    method_id   = method_data["job_tag"]

    try:
        spark  = SparkConfig().spark_session("APP_CFG", "query_dicom")
        df = spark.read.format("delta").load(method_data['table_path'])\
            .where(f"metadata.SeriesInstanceUID='{datastore.address}'")\
            .where( "metadata.InstanceNumber='1'")\
            .select("path", "metadata")\
            .toPandas()
        
        spark.stop()
        
        if not len(df) == 1: raise ValueError(f"Resulting query record is not singular, multiple scan's exist given the container address {datastore.address}")
            
        record = df.loc[0]
    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e
    else:
        dicom = Node("DicomSeries", method_id, record['metadata'])
        # Do some path processing...we pulled the first dicom image, but need the parent image folder
        data_path = Path(record['path'].split(':')[-1]).parent
        dicom.set_data(data_path)
        datastore.put(dicom)
        
    
if __name__ == "__main__":
    cli()
