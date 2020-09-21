import os
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]='/gpfs/mskmindhdp_emc/sw/env/bin/python' 
os.environ["PYSPARK_DRIVER_PYTHON"]='/gpfs/mskmindhdp_emc/sw/env/bin/python' 

"""Common spark session"""
class SparkConfig:

	def spark_session(self, app_name, spark_uri, spark_driver_host="127.0.0.1"):
		return SparkSession.builder \
			.appName(app_name) \
			.master(spark_uri) \
			.config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
			.config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore") \
			.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
			.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
			.config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
			.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
			.config("spark.executor.memory", "6g") \
			.config("spark.driver.memory", "6g") \
			.config("fs.defaultFS", "file:///") \
			.config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
			.getOrCreate()
							
