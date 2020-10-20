'''
Created on September 11, 2020

@author: pashaa@mskcc.org

This module generates a delta table for clinical data stored in a csv or tsv file with tab delimiters.
'''
import click
from data_processing.common.custom_logger import init_logger
from pyspark.sql.functions import expr
from pyspark.sql.utils import AnalysisException

from data_processing.common.sparksession import SparkConfig

logger = init_logger()


def generate_proxy_table(source_file, destination_dir, spark):

    logger.info("generating proxy table...")
    try:
        df = spark.read.options(header='True', inferSchema='True', delimiter='\t').csv(source_file)
        df = df.withColumn("uuid", expr("uuid()"))
        df.coalesce(128).write.format("delta"). \
            mode('overwrite'). \
            save(destination_dir)
    except AnalysisException:
        raise Exception("Make sure input file is a valid tab delimited csv file")

    df.printSchema()
    df.show()


@click.command()
@click.option('-s', '--source_file',
              type=click.Path(exists=True),
              default = None,
              help = "file path to source tsv or csv file")
@click.option('-d', '--destination_dir',
              type=click.Path(exists=False),
              default="./clinical_delta_table",
              help="directory path to the clinical delta table")
@click.option('-m', '--spark_master_uri',
              help='spark master uri e.g. spark://master-ip:7077 or local[*]',
              required=True)
@click.option('-f', '--config_file', default = 'config.yaml', help="config file")
def cli(spark_master_uri, source_file, destination_dir, config_file):
    """
    This module generates a delta table for clinical data stored in a csv or tsv file with tab delimiters.

    Example:
        python src/clinical_proxy_tables.py \
                 --spark_master_uri {spark_master_uri} \
                 --source_file <path to csv file> \
                 --destination_path <path to delta_table dir>
    """
    # Setup Spark context
    import time
    start_time = time.time()
    spark = SparkConfig().spark_session(config_file, "clinical-proxy-preprocessing")
    generate_proxy_table(source_file, destination_dir, spark)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    cli()
