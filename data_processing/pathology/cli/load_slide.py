'''
Created: February 2021
@author: aukermaa@mskcc.org

Given a scan (container) ID
1. load WholeSlideImage object given a delta table path 

'''

# General imports
import os, json, logging
import click
from pathlib import Path

# From common
from data_processing.common.custom_logger   import init_logger
from data_processing.common.DataStore       import DataStore
from data_processing.common.Node            import Node
from data_processing.common.config          import ConfigSet
from data_processing.common.sparksession     import SparkConfig


@click.command()
@click.option('-a', '--app_config', required=True,
              help="application configuration yaml file. See config.yaml.template for details.")
@click.option('-c', '--cohort_id', required=True,
              help="cohort name")
@click.option('-s', '--datastore_id', required=True,
              help='datastore name. usually a slide id.')
@click.option('-m', '--method_param_path', required=True,
              help='json parameter file with path to a WSI delta table.')

def cli(app_config, cohort_id, datastore_id, method_param_path):
    init_logger()

    with open(method_param_path) as json_file:
        method_data = json.load(json_file)
    load_slide_with_datastore(app_config, cohort_id, datastore_id, method_data)

def load_slide_with_datastore(app_config, cohort_id, container_id, method_data):
    """
    Using the container API interface, fill scan with original slide from table
    """
    logger = logging.getLogger(f"[datastore={container_id}]")

    # Do some setup
    cfg = ConfigSet("APP_CFG", config_file=app_config)
    datastore = DataStore(cfg)
    datastore.createNamespace(cohort_id)
    datastore.setNamespace(cohort_id)
    datastore.createDatastore(container_id, 'slide')
    datastore.setDatastore  (container_id)
    method_id   = method_data["job_tag"]

    # fetch patient_id column 
    patient_id_column  = method_data.get("patient_id_column_name", None)
    if patient_id_column == "": patient_id_column = None

    try:
        spark  = SparkConfig().spark_session("APP_CFG", "query_slide")
        slide_id = datastore._name.upper()

        if patient_id_column:
            # assumes parquet from dremio for now, todo: change
            df = spark.read.parquet(method_data['table_path'])\
                .where(f"UPPER(slide_id)='{slide_id}'")\
                .select("path", "metadata", patient_id_column)\
                .toPandas()

        else:
            df = spark.read.format("delta").load(method_data['table_path'])\
                .where(f"UPPER(slide_id)='{slide_id}'")\
                .select("path", "metadata")\
                .toPandas()

        spark.stop()
        
        if not len(df) == 1: 
            print(df)
            raise ValueError(f"Resulting query record is not singular, multiple scan's exist given the container address {slide_id}")
            
        record = df.loc[0]

    except Exception as e:
        logger.exception (f"{e}, stopping job execution...")
        raise e

    # convert nested row-type into dict
    metadata = record['metadata'][0]
    properties = {x.asDict()['key']:x.asDict()['value'] for x in metadata}
    if patient_id_column:
        properties['patient_id'] = record[patient_id_column]


    print(properties)
    # Put results in the data store
    slide = Node("WholeSlideImage", method_id, properties)
    # Do some path processing...we pulled the first dicom image, but need the parent image folder
    data_path = Path(record['path'].split(':')[-1])
    slide.set_data(data_path)
    datastore.put(slide)
        
    
if __name__ == "__main__":
    cli()
