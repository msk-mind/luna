'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. resolve the path to the dicom folder
2. for all dicoms, rescale into HU and optionally window
3. store results on HDFS and add metadata to the graph

'''

# General imports
import os, json, sys
import click
from pathlib import Path

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.Container       import Container
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet

# From common
from data_processing.common.sparksession     import SparkConfig


logger = init_logger("get_dicoms.log")
cfg = ConfigSet("APP_CFG",  config_file="config.yaml")

@click.command()
@click.option('-c', '--cohort_id',    required=True)
@click.option('-s', '--container_id', required=True)
@click.option('-m', '--method_param_path',    required=True)
def cli(cohort_id, container_id, method_param_path):
    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    load_dicom_with_container(cohort_id, container_id, method_data)

def load_dicom_with_container(cohort_id, container_id, method_data):
    """
    Using the container API interface, fill scan with original dicoms from table
    """
    # Do some setup
    container   = Container( cfg ).setNamespace(cohort_id).lookupAndAttach(container_id)
    method_id   = method_data["job_tag"]

    try:
        spark  = SparkConfig().spark_session("APP_CFG", "query_dicom")
        df = spark.read.format("delta").load(method_data['table_path'])\
            .where(f"metadata.SeriesInstanceUID='{container.address}'")\
            .where( "metadata.InstanceNumber='1'")\
            .select("path", "metadata")\
            .toPandas()
        
        spark.stop()
        
        if not len(df) == 1: raise ValueError(f"Resulting query record is not singular, multiple scan's exist given the container address {container.address}")
            
        record = df.loc[0]
    except Exception as e:
        container.logger.exception (f"{e}, stopping job execution...")
    else:
        dicom = Node("DicomSeries", method_id, record['metadata'])
        # Do some path processing...we pulled the first dicom image, but need the parent image folder
        data_path = Path(record['path'].split(':')[-1]).parent
        dicom.set_data(data_path)
        container.add(dicom)
        container.saveAll()
    
if __name__ == "__main__":
    cli()
