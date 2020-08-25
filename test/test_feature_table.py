import os, click, shutil
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/') ))
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../dl_processing/') ))

from sparksession import SparkConfig
import numpy as np
from unittest import TestCase, mock
import pandas as pd
from joblib import Parallel, delayed
from medpy.io import load
from pyspark import *
from skimage.transform import resize
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType
from preprocess_feature import generate_feature_table, cli
from click.testing import CliRunner

def test_local_feature_table(sc):
    import time

    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, sc,False)

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/feature_table")
    
    # Read Delta Table and Verify 
    feature_table = DeltaTable.forPath(sc, feature_table_path)
    feature_df = feature_table.toDF()
    assert feature_table.toDF().count() == 15
    print ("Test Local feature table passed.")
    feature_dir = os.path.join(BASE_DIR[7:], "features")
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir) 


def test_feature_table_cli(runner):
    test_args = ["--spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables --target_spacing 1.0 1.0 3.0",
            "--spark_master_uri local[*] --base_directory /gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables --target_spacing 1 1 3"]

    for arg in test_args:
        result = runner.invoke(cli, arg)
        assert result.exit_code == 0
        feature_dir = os.path.join(BASE_DIR[7:], "features")
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir) 

    print("Test CLI arguments tests passed.")

sc = SparkConfig().spark_session("test-feature-table-generation", "local[*]")

# Need to add python dependencies to the spark context, otherwise get module not found error
sc.sparkContext.addPyFile("/gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/dl_processing/preprocess_feature.py")
sc.sparkContext.addPyFile("/gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/src/sparksession.py")

from delta.tables import DeltaTable
BASE_DIR = "file:///gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables/"
TARGET_SPACING = (1.0, 1.0, 3.0)

# Test Local Feature Table
test_local_feature_table(sc)

# Test CLI parameters
runner = CliRunner()
test_feature_table_cli(runner)


#Test HDFS Feature Table Generation (todo)

