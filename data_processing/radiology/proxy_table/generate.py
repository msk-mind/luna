'''
Created on October 29, 2020

@author: pashaa@mskcc.org
'''
import click

from data_processing.common.sparksession import SparkConfig


@click.command()
@click.option('-t', '--template', default=None,
              help="path to yaml template file containing information required for radiology proxy data ingestion. "
                   "See data_processing/radiology/proxy_table/data_ingestion_template.yaml.template")
@click.option('-f', '--config_file', default='config.yaml',
              help="path to config file containing application configuration. See config.yaml.template")
def cli(template,
        config_file):
    """
    This module generates a set of proxy tables for radiology data based on information specified in the tempalte file.

    Example:
        python -m data_processing.radiology.proxy_table.generate \
        --template {PATH_TO_TEMPLATE_FILE} \
        --config_file {PATH_TO_CONFIG_FILE}
    """
    # Setup Spark context
    import time
    start_time = time.time()


    #spark = SparkConfig().spark_session(config_file, "data_processing.radiology.proxy_table.generate")
    print(template+' '+config_file)
    print("--- Finished in %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    cli()