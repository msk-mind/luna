from pyspark.sql import SparkSession
import sys
import os
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../') ))

from sparksession import SparkConfig

def write_delta_table(tsv_path, table_path):
	df = spark.read.csv(tsv_path, sep=r'\t', header=True)
	df.write.format("delta").mode("overwrite").save(table_path)

	from delta.tables import DeltaTable
	dt = DeltaTable.forPath(spark, table_path)

spark = SparkConfig().spark_session("create-mock-scan-annotation", "local[*]")

annot_manifest_path = "file:///gpfs/mskmind_ess/eng/radiology/testdata/annotation_manifest_ovarian.tsv"
annot_table_path = "file:///gpfs/mskmind_ess/eng/radiology/testdata/tables/annotation"

scan_manifest_path = "file:///gpfs/mskmind_ess/eng/radiology/testdata/scan_manifest_ovarian.tsv"
scan_table_path = "file:///gpfs/mskmind_ess/eng/radiology/testdata/tables/scan"

write_delta_table(annot_manifest_path, annot_table_path)
write_delta_table(scan_manifest_path, scan_table_path)
