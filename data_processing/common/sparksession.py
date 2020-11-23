import sys
import os

sys.path.append(os.path.abspath('../'))
	
from pyspark.sql import SparkSession
from common.config import Config

"""Common spark session"""
class SparkConfig:

	def spark_session(self, config_file, app_name):


		config = Config(config_file=config_file)

		spark_uri = config.get_value('$.spark_cluster_config[:1]["spark.uri"]')
		spark_driver_host = config.get_value('$.spark_cluster_config[:2]["spark.driver.host"]')
		spark_executor_cores = config.get_value('$.spark_application_config[:1]["spark.executor.cores"]')
		spark_cores_max = config.get_value('$.spark_application_config[:2]["spark.cores.max"]')
		spark_executor_memory = config.get_value('$.spark_application_config[:3]["spark.executor.memory"]')
		spark_executor_pyspark_memory = \
			config.get_value('$.spark_application_config[:4]["spark.executor.pyspark.memory"]')
		spark_sql_shuffle_partitions = config.get_value('$.spark_application_config[:5]["spark.sql.shuffle.partitions"]')

		return SparkSession.builder \
			.appName(app_name) \
			.master(spark_uri) \
			.config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
			.config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore") \
			.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
			.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
			.config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
			.config("spark.driver.host", spark_driver_host) \
			.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
			.config("spark.executor.memory", spark_executor_memory) \
			.config("spark.driver.memory", spark_executor_memory) \
			.config("spark.executor.cores", spark_executor_cores) \
			.config("spark.cores.max", spark_cores_max) \
			.config("spark.executor.pyspark.memory", spark_executor_pyspark_memory) \
			.config("spark.sql.shuffle.partitions", spark_sql_shuffle_partitions) \
			.config("fs.defaultFS", "file:///") \
			.config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.getOrCreate()
							
