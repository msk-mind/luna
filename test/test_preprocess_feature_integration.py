import os, click, shutil
import sys
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/') ))
sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../src/dl_processing/') ))

from sparksession import SparkConfig
import pytest
from preprocess_feature import cli, generate_feature_table
from click.testing import CliRunner

"""
To run the test,

cd data-processing
pytest -s --cov=src test
"""
#BASE_DIR = "/gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables/"
BASE_DIR = "/gpfs/mskmind_ess/rosed2"
TARGET_SPACING = (1.0, 1.0, 3.0)

# Test CLI parameters
runner = CliRunner()

#TODO write tests with Mock
#TODO increase test coverage
#TODO Test HDFS Feature Table Generation

@pytest.fixture(autouse=True)
def cleanup():
    feature_dir = os.path.join(BASE_DIR, "features")
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)

def test_local_feature_table_generation(spark_session):
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")

    # Test no query, default naming
    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "subtype='BRCA1' or subtype='BRCA2'", "feature-table-test-name")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/feature-table-test-name")
    
    # Read Delta Table and Verify 
    feature_df = spark_session.read.format("delta").load(feature_table_path)
    assert feature_df.count() == 11
    print ("test_local_feature_table_generation passed.")


def test_local_feature_table_generation_malformed_query(spark_session):
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")

    # Test no query, default naming
    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "subtype='BRCA1' || subbtype='BRCA2'", "")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/feature-table-test-name")
    
    assert not os.path.exists(feature_table_path)
    print ("test_local_feature_table_generation_malformed_query passed.")


def test_feature_table_cli():
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3 --query \"subtype=\'CCNE1\'\" --feature_table_output_name test-cli-2".format(BASE_DIR))
    assert result.exit_code == 0
    print("CLI test passed.")


def test_feature_table_cli_missing_params():
    test_args = ["--spark_master_uri local[*] --target_spacing 1.0 1.0 3.0",
            "--base_directory {} --target_spacing 1 1 3".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {}".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {} --target_spacing 1.0 1.0 3.0".format("/path/does/not/exist")]

    for arg in test_args:
        result = runner.invoke(cli, arg)
        assert result.exit_code == 2

    print("Test CLI missing arguments tests passed.")

