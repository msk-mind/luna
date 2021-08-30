"""
This module is a wrapper around luna/radiology/refined_table/generate.py

It takes a list of patient ids and looks up SeriesInstanceUIDs tied to the patient ids.
Then it calls the generate.py with a SeriesInstanceUID to generate a scan.
"""
import click
import os

from luna.common.CodeTimer import CodeTimer
from luna.common.Neo4jConnection import Neo4jConnection
from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.common.custom_logger import init_logger
import luna.common.constants as const
from luna.radiology.refined_table.generate import generate_scan_table

logger = init_logger()
logger.info("Starting scan generation cli.py")


def validate_file_ext(ctx, param, value):
    if not value in ['mhd','nrrd']:
        raise click.UsageError("file_ext should be one of mhd|nrrd")
    else:
        return value.lower()


@click.command()
@click.option('-a', '--app_config_file', default = 'config.yaml', help="application config file")
@click.option('-c', '--package_config_file', default = 'luna/radiology/refined_table/config.yaml',
	help="package specific config. See config.yaml.template")
@click.option('-f', '--patient_file', type=click.Path(exists=True), help = "A file containing one patient id per line.")
@click.option('-i', '--patient_id_type', help = "A file containing one patient id per line.")
@click.option('-t', '--tag', default = 'default', help="Provencence tag")
@click.option('-p', '--project_name', help="MIND project address")
@click.option('-e', '--file_ext', callback=validate_file_ext, help="file format for scan generation", required=True)
@click.option('-s', '--custom_preprocessing_script', default = 'luna/radiology/refined_table/dicom_to_scan.py',
	help="Path to python file to execute in the working directory", required=True)
def cli(app_config_file, package_config_file, patient_file, patient_id_type, tag, project_name, file_ext, custom_preprocessing_script):
	"""
	Usage:
		python3 -m luna.radiology.refined_table.cli \
			\
			\
	"""
	# load configurations
	cfg = ConfigSet(name=const.APP_CFG, config_file=app_config_file)
	cfg = ConfigSet(name=const.DATA_CFG, config_file=package_config_file)

	spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="scan-cli")

	# load patient id file into a dataframe
	pid = spark.read.text(patient_file) # value column


	# set up graph connection
	conn = Neo4jConnection(uri=cfg.get_value(name=const.DATA_CFG, jsonpath='GRAPH_URI'),
		user=cfg.get_value(name=const.DATA_CFG, jsonpath='GRAPH_USER'),
		pwd=cfg.get_value(name=const.DATA_CFG, jsonpath='GRAPH_PW'))

	def call_generate_scan(df: pd.DataFrame) -> pd.DataFrame:
		# match scan where dmp_patient_id=pid
		#conn.query()

		# update all SIUID loop!
		
		# generate_scan_table(spark, uid, hdfs_uri, custom_preprocessing_script, tag, project_name, file_ext):


	df = pid.groupby("value").applyInPandas(call_generate_scan, schema=pid.schema)

	query = f"MATCH (patient:{patient_id_type})-[:PX_TO_RAD]-(rad)-[:HAS_SCAN]-(scan) WHERE patient.value='{pid}' RETURN patient, scan"
	seriesInstanceUIDs = [ x.data()['scan']['SeriesInstanceUID'] for x in conn.query(query) ]

