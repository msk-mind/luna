from pyspark.sql import SparkSession
from luna.common.config import ConfigSet
import warnings
"""Common spark session"""
class SparkConfig:

	def spark_session(self, config_name, app_name):
		'''
		@:param config_name logical name of the configuration to use from the ConfigSet. See luna/common/config.py
		@:param app_name application name to give to the spark session. This name will be used in the spark logs.
		'''

		cfg = ConfigSet()
		warnings.warn("You are using SparkConfig to generate a spark_session, however spark has been depreciated from this package!")

		spark_uri = cfg.get_value(path=config_name+'::$.spark_cluster_config[:1]["spark.uri"]')
		spark_driver_host = cfg.get_value(path=config_name+'::$.spark_cluster_config[:2]["spark.driver.host"]')
		spark_executor_cores = cfg.get_value(path=config_name+
												  '::$.spark_application_config[:1]["spark.executor.cores"]')
		spark_cores_max = cfg.get_value(path=config_name+'::$.spark_application_config[:2]["spark.cores.max"]')
		spark_executor_memory = cfg.get_value(path=config_name+
												   '::$.spark_application_config[:3]["spark.executor.memory"]')
		spark_executor_pyspark_memory = \
			cfg.get_value(path=config_name+'::$.spark_application_config[:4]["spark.executor.pyspark.memory"]')
		spark_sql_shuffle_partitions = \
			cfg.get_value(path=config_name+'::$.spark_application_config[:5]["spark.sql.shuffle.partitions"]')
		spark_driver_maxresultsize = \
			cfg.get_value(path=config_name+'::$.spark_application_config[:6]["spark.driver.maxResultSize"]')

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
			.config("spark.driver.maxResultSize", spark_driver_maxresultsize) \
			.config("spark.files.overwrite", "true") \
			.config("fs.defaultFS", "file:///") \
			.config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.getOrCreate()

