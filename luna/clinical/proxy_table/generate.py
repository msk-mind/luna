'''
Created on September 11, 2020

@author: pashaa@mskcc.org

This module generates a delta table for clinical data stored in a csv or tsv file with tab delimiters.
'''
import click
import os, shutil

from luna.common.CodeTimer import CodeTimer
from luna.common.config import ConfigSet
from luna.common.custom_logger import init_logger

from pyspark.sql.functions import expr
from pyspark.sql.utils import AnalysisException

from luna.common.sparksession import SparkConfig
import luna.common.constants as const

logger = init_logger()

def generate_proxy_table():

    cfg = ConfigSet()
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="luna.clinical.proxy_table.generate")

    logger.info("generating proxy table...")
    source_path = cfg.get_value(path=const.DATA_CFG+'::SOURCE_PATH')
    file_ext = cfg.get_value(path=const.DATA_CFG+'::FILE_TYPE')

    delimiter = ''
    if file_ext and file_ext.lower() == 'csv':
        delimiter = ','
    elif file_ext and file_ext.lower() == 'tsv':
        delimiter = '\t'
    else:
        raise Exception("Make sure input file is a valid tsv or csv file")

    table_location = const.TABLE_LOCATION(cfg)

    df = spark.read.options(header='True', inferSchema='True', delimiter=delimiter).csv(source_path)
    # generate uuid
    df = df.withColumn("uuid", expr("uuid()"))

    df.coalesce(cfg.get_value(path=const.DATA_CFG+'::NUM_PARTITION')).write.format("delta"). \
        mode('overwrite'). \
        save(table_location)

    df.printSchema()
    df.show()


@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing data input and output parameters. "
                   "See ./data_config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to yaml file containing application runtime parameters. "
                   "See ./app_config.yaml.template")
def cli(data_config_file, app_config_file):
    """
    This module generates a delta table with clinical data based on the input and output parameters specified in
     the data_config_file.

    Example:
        python3 -m luna.clinical.proxy_table.generate \
                 --data_config_file <path to data config file> \
                 --app_config_file <path to app config file>
    """
    with CodeTimer(logger, 'generate clinical proxy table'):
        # Setup configs
        cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
        # data_type used to build the table name can be pretty arbitrary, so left the schema file out for now.
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)

        # copy app and data configuration to destination config dir
        config_location = const.CONFIG_LOCATION(cfg)
        os.makedirs(config_location, exist_ok=True)

        shutil.copy(app_config_file, os.path.join(config_location, "app_config.yaml"))
        shutil.copy(data_config_file, os.path.join(config_location, "data_config.yaml"))
        logger.info("config files copied to %s", config_location)

        generate_proxy_table()


if __name__ == "__main__":
    cli()
