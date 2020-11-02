import click
from pyspark.sql import SparkSession
from data_processing.common.sparksession import SparkConfig
from data_processing.common.custom_logger import init_logger
import re, os

# Run Script: python3 dicom_to_delta.py -h <hdfs> -a <radiology_dataset_path>
log = init_logger()

def create_delta_table(df, table_name, delta_path, merge, purge):
	"""
	Create delta table from the dataframe.
	"""
	# TODO repartition # based on the number of rows/performance
	# TODO upsert - do we still need merge?
	#   dcm: update on match AccessionNumber|AcquisitionNumber|SeriesNumber|InstanceNumber, otherwise insert
	#   binary: ...later for when we embed it in parquet
	#   op: update on match filename, otherwise insert

	if purge:
		# Changing a column's type or name or dropping a column requires rewriting the table.
		df.coalesce(128) \
			.write \
			.format("delta") \
			.option("overwriteSchema", "true") \
			.mode("overwrite") \
			.save(delta_path)
	if merge:
		# Adding new columns can be achived with .option("mergeSchema", "true")
		df.coalesce(128) \
			.write \
			.format("delta") \
			.option("mergeSchema", "true") \
			.mode("append") \
			.save(delta_path)

	else:
		df.coalesce(128) \
			.write \
			.format("delta") \
			.mode("append") \
			.save(delta_path)



def remove_delta_table(spark, delta_path):
	"""
	Clean up an existing delta table.
	"""
	from delta.tables import DeltaTable

	dt = DeltaTable.forPath(spark, delta_path)

	# Disable check for retention - default 136 hours (7 days)
	spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
	dt.vacuum(0) # vacuum all data


@click.command()
@click.option('-f', '--config_file', default = 'config.yaml', help="config file")
@click.option("-h", "--hdfs", default="hdfs://sandbox-hdp.hortonworks.com:8020/", show_default=True, help="HDFS uri e.g. hdfs://sandbox-hdp.hortonworks.com:8020/")
@click.option("-a", "--dataset_address", required=True, help= "path to dataset directory containing parquet/ and dicom/ directories. This path is the directory for which the new delta table directory /table is created under")
# Note: use with caution. Either use merge or purge, not both.
@click.option("-m", "--merge", is_flag=True, default=False, show_default=True, help="(optional) Merge schema - add new columns")
@click.option("-p", "--purge", is_flag=True, default=False, show_default=True, help="(optional) Delete all delta tables - then create new tables")
def cli(config_file, hdfs ,dataset_address, merge, purge):
	"""
	Main CLI - setup spark session and call write to delta.
	"""
	if merge and purge:
		raise ValueError("Cannot use flags merge and purge at the same time!")

	sc = SparkConfig()
	spark_session = sc.spark_session(config_file, "dicom-to-delta")

	write_to_delta(spark_session, hdfs , dataset_address, merge, purge)


def write_to_delta(spark, hdfs, dataset_address, merge, purge):
	"""
	Create proxy tables from Dicom parquet files at {dataset_address} directory
	"""

	# form dataset addresss (contains /parquet, /table, /dicom, and dataset yaml)
	# in case dataset_address is an absolute path, we use + instead of os.path.join
	dataset_address = hdfs + "/" + dataset_address
	# input parquet file paths:
	# path to dicom binary parquet
	binary_path = os.path.join(dataset_address, "dicom")
	# path to parquet containing dicom headers and op metadata
	dcm_path = os.path.join(dataset_address, "parquet")
	# output delta table names:
	binary_table = "dicom_binary"
	dcm_table = "dicom"

	# output delta table paths
	common_delta_path = os.path.join(dataset_address, "table")
	binary_delta_path = os.path.join(common_delta_path, binary_table)
	dcm_delta_path = os.path.join(common_delta_path, dcm_table)

	# Read dicom binary files
	binary_df = spark.read.format("binaryFile") \
		.option("pathGlobFilter", "*.dcm") \
		.option("recursiveFileLookup", "true") \
		.load(binary_path)
	# Read parquet files containing dicom header and dicom op metadata
	dcm_df = spark.read.parquet(dcm_path)

	# To improve read performance when you load data back, Databricks recommends
	# turning off compression when you save data loaded from binary files:
	spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

	# clean up the latest delta table version
	if purge:
		log.info("Purging dicom and dicom_binary tables..")
		remove_delta_table(spark, binary_delta_path)
		remove_delta_table(spark, dcm_delta_path)

	# Create Delta tables
	create_delta_table(binary_df, binary_table, binary_delta_path, merge, purge)
	create_delta_table(dcm_df, dcm_table, dcm_delta_path, merge, purge)
	log.info("Created dicom and dicom_binary tables")


if __name__ == '__main__':
	cli()
