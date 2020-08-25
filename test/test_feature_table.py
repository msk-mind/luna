import os, click
import sys, pytest
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/') ))
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../dl_processing/') ))

from sparksession import SparkConfig
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from skimage.transform import resize
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType
from  preprocess_feature import generate_feature_table
from click.testing import CliRunner


def test_local_feature_table(runner):
    import time

    # Build Feature Table
    result = runner.invoke(generate_feature_table, "--spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables --target_spacing 1.0 1.0 3.0")
    # generate_feature_table('local[*]', BASE_DIR, str(TARGET_SPACING), False)

    # read and verify correct feature table generated
    # os.sys("python dl_processing/preprocess_feature.py --spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables --target_spacing 1.0 1.0 3.0")
    feature_table_path = os.path.join(BASE_DIR, "features/feature_table")
    
    # Read Delta Table and Verify 
    # feature_table = DeltaTable.forPath(spark, feature_table_path)
    feature_table = spark.read.format("delta").load(feature_table_path)
    assert feature_table.toDF().count() == 15
    print ("Test Local feature table passed.")



spark = SparkConfig().spark_session("test-feature-table-generation", "local[*]")
from delta.tables import DeltaTable
BASE_DIR = "file:////gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables/"
TARGET_SPACING = (1.0, 1.0, 3.0)
runner = CliRunner()
test_local_feature_table(runner)

# dl_preprocessing command
# python preprocess_feature.py --spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables --target_spacing 1.0 1.0 3.0