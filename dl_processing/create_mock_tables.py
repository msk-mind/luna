from pyspark.sql import SparkSession


spark = SparkSession.builder \
	.appName("create-mock-scan-annotation") \
	.config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
	.config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore") \
	.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
	.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
	.getOrCreate()

def write_delta_table(csv_path, table_path):
	df = spark.read.format("csv").option("header", "true").load(csv_path)
	df.write.format("delta").mode("overwrite").save(table_path)

	from delta.tables import DeltaTable
	dt = DeltaTable.forPath(spark, table_path)


annot_csv_path = "file:///home/rosed2/annotation_mock.csv"
annot_table_path = "file:///gpfs/mskmind_ess/rosed2/tables/annotation"

scan_csv_path = "file:///home/rosed2/scan_mock.csv"
scan_table_path = "file:///gpfs/mskmind_ess/rosed2/tables/scan"

write_delta_table(annot_csv_path, annot_table_path)
write_delta_table(scan_csv_path, scan_table_path)
