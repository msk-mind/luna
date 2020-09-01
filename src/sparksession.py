from pyspark.sql import SparkSession

"""Common spark session"""

class SparkConfig:

	def spark_session(self, app_name, spark_master_uri, hdfs=False):

		default_fs = "file:///"
		if hdfs:
			default_fs = "hdfs://"
			
		return SparkSession.builder \
			.appName(app_name) \
			.master(spark_master_uri) \
			.config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
			.config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore") \
			.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
			.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
			.config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
			.config("spark.driver.port", "8001") \
			.config("spark.blockManager.port", "8002") \
			.config("spark.driver.bindAddress", "0.0.0.0") \
			.config("fs.defaultFS", default_fs) \
			.getOrCreate()
