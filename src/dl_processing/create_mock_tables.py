from pyspark.sql import SparkSession
import sys
import os
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../') ))

from sparksession import SparkConfig

def write_delta_table(csv_path, table_path):
	df = spark.read.format("csv").option("header", "true").load(csv_path)
	df.write.format("delta").mode("overwrite").save(table_path)

	from delta.tables import DeltaTable
	dt = DeltaTable.forPath(spark, table_path)

spark = SparkConfig().spark_session("create-mock-scan-annotation", "local[*]")

annot_csv_path = "file:///Users/rosed2/Downloads/annotation_mock.csv"
annot_table_path = "file:///Users/rosed2/Downloads/tables/annotation"

scan_csv_path = "file:///Users/rosed2/Downloads/scan_mock.csv"
scan_table_path = "file:///Users/rosed2/Downloads/tables/scan"

write_delta_table(annot_csv_path, annot_table_path)
write_delta_table(scan_csv_path, scan_table_path)
