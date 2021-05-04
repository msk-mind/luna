'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. load WholeSlideImage object given a delta table path 

'''

# General imports
import os, json, sys
import click
from pathlib import Path

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet
from data_processing.common.sparksession     import SparkConfig


logger = init_logger("load_slide.log")

@click.command()
@click.option('-a', '--app_config', required=True)
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--datastore_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(app_config, cohort_id, datastore_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    load_slide_with_datastore(app_config, cohort_id, datastore_id, method_data)

def load_slide_with_datastore(app_config, cohort_id, container_id, method_data):
    """
    Using the container API interface, fill scan with original slide from table
    """
    # Do some setup
    cfg = ConfigSet("APP_CFG",  config_file=app_config)
    datastore   = DataStore( cfg ).setNamespace(cohort_id).setDatastore(container_id)
    method_id   = method_data["job_tag"]

    try:
        spark  = SparkConfig().spark_session("APP_CFG", "query_slide")
        df = spark.read.format("delta").load(method_data['table_path'])\
            .where(f"slide_id='{datastore.address}'")\
            .select("path", "metadata")\
            .toPandas()

        spark.stop()
        
        if not len(df) == 1: raise ValueError(f"Resulting query record is not singular, multiple scan's exist given the container address {datastore.address}")
            
        record = df.loc[0]
    except Exception as e:
        datastore.logger.exception (f"{e}, stopping job execution...")
        return

    # Put results in the data store
    slide = Node("WholeSlideImage", method_id, record['metadata'])
    # Do some path processing...we pulled the first dicom image, but need the parent image folder
    data_path = Path(record['path'].split(':')[-1])
    slide.set_data(data_path)
    datastore.put(slide)
        
    
if __name__ == "__main__":
    cli()
