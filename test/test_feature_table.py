import os, click, shutil
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/') ))
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/dl_processing/') ))

from sparksession import SparkConfig
from unittest import TestCase, mock
import pytest
from preprocess_feature import cli, generate_feature_table
from click.testing import CliRunner

"""
To run the test,

cd data-processing
pytest -s --cov=dl_processing test
"""
BASE_DIR = "/gpfs/mskmind_ess/rosed2/"
TARGET_SPACING = (1.0, 1.0, 3.0)

# Test CLI parameters
runner = CliRunner()

#TODO try spark fixture: https://github.com/malexer/pytest-spark
#TODO write tests with Mock
#TODO increase test coverage
#TODO Test HDFS Feature Table Generation

def test_local_feature_table():
    sc = SparkConfig().spark_session("test-feature-table-generation", "local[*]")
    from delta.tables import DeltaTable
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    sc.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    sc.sparkContext.addPyFile("src/sparksession.py")

    # Build Feature Table
    generate_feature_table("file:///"+BASE_DIR, TARGET_SPACING, sc, False)

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


def test_feature_table_cli():
    test_args = ["--spark_master_uri local[*] --base_directory {} --target_spacing 1.0 1.0 3.0".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3".format(BASE_DIR)]

    for arg in test_args:
        result = runner.invoke(cli, arg)
        assert result.exit_code == 0
        feature_dir = os.path.join(BASE_DIR, "features")
        if os.path.exists(feature_dir):
            shutil.rmtree(feature_dir) 

    print("Test CLI arguments tests passed.")


