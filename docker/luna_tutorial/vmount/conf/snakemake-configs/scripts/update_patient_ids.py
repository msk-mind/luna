from pyspark.sql import SparkSession
import shutil

# ------------------------------------------------------------------------------------------------------------------------------------------
# updates the wsi table to include patient_ids

wsi_table_path = "/gpfs/mskmindhdp_emc/user/shared_data_folder/pathology-tutorial/PRO_12-123/data/PRO_12-123/tables/WSI_toy_data_set"

# setup spark session
spark = SparkSession.builder \
        .appName("test") \
        .master('local[*]') \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.7.0") \
        .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .getOrCreate()

# read WSI delta table
wsi_table = spark.read.format("delta").load("file:///" + wsi_table_path).toPandas()

# remove current WSI toy dataset before replacing
shutil.rmtree(wsi_table_path)

# insert spoof patient ids
patient_id=[1,2,3,4,5]
wsi_table["patient_id"]=patient_id

# convert back to a spark table (update table)
x = spark.createDataFrame(wsi_table)
x.write.format("delta").mode("overwrite").option("mergeSchema", "true").save("file:///" + wsi_table_path)