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
BASE_DIR = "/gpfs/mskmind_ess/pateld6/work/sandbox/data-processing/test-tables/"
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

def test_local_feature_table(spark_session):
    from delta.tables import DeltaTable
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")


    # Test no query, default naming
    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "", "feature-table")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/feature-table")
    
    # Read Delta Table and Verify 
    feature_table = DeltaTable.forPath(spark_session, feature_table_path)
    feature_df = feature_table.toDF()
    assert feature_table.toDF().count() == 15
    print ("test_local_feature_table passed.")

def test_local_feature_table_query(spark_session):
    from delta.tables import DeltaTable
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")


    # Test 2: SQL query test 
    # Build Feature Table 
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "subtype='BRCA1' or subtype='BRCA2'", "feature-table")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/feature-table")
    
    # Read Delta Table and Verify 
    feature_table = DeltaTable.forPath(spark_session, feature_table_path)
    feature_df = feature_table.toDF()
    assert feature_table.toDF().count() == 11
    print ("test_local_feature_table_query passed.")

def test_local_feature_table_name(spark_session):
    from delta.tables import DeltaTable
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")

    # Test 3: test feature table naming 
    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "", "test-name")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/test-name")
    
    # Read Delta Table and Verify 
    feature_table = DeltaTable.forPath(spark_session, feature_table_path)
    feature_df = feature_table.toDF()
    assert feature_table.toDF().count() == 15
    print ("test_local_feature_table_name passed.")

def test_local_feature_table_name_and_query(spark_session):
    from delta.tables import DeltaTable
    
    # Need to add python dependencies to the spark context, otherwise get module not found error
    spark_session.sparkContext.addPyFile("src/dl_processing/preprocess_feature.py")
    spark_session.sparkContext.addPyFile("src/sparksession.py")

    # Test 4: test feature table naming and filtering 
    # Build Feature Table
    generate_feature_table(BASE_DIR, TARGET_SPACING, spark_session, False, "", "test-name-two")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "features/test-name-two")
    
    # Read Delta Table and Verify 
    feature_table = DeltaTable.forPath(spark_session, feature_table_path)
    feature_df = feature_table.toDF()
    assert feature_table.toDF().count() == 15
    print ("test_local_feature_table_name_and_query passed.")


def test_feature_table_cli():
    
    # Test 1, no query or naming
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3".format(BASE_DIR))
    assert result.exit_code == 0
    feature_dir = os.path.join(BASE_DIR, "features")
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)
    print("CLI Test 1 passed.")

    # Test 2, query 
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3 --query \"annotation_uid=\'annotation-1\'\"".format(BASE_DIR))
    assert result.exit_code == 0
    print("CLI Test 2 passed.")

    # Test 3, name
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3 --feature_table_output_name test-cli-1".format(BASE_DIR))
    assert result.exit_code == 0
    print("CLI Test 3 passed.")

    # Test 4, query and name
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3 --query \"subtype=\'CCNE1\'\" --feature_table_output_name test-cli-2".format(BASE_DIR))
    assert result.exit_code == 0
    print("CLI Test 4 passed.")

    print("All CLI tests passed.")


def test_feature_table_cli_missing_params():
    test_args = ["--spark_master_uri local[*] --target_spacing 1.0 1.0 3.0",
            "--base_directory {} --target_spacing 1 1 3".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {}".format(BASE_DIR)]

    for arg in test_args:
        result = runner.invoke(cli, arg)
        assert result.exit_code == 2

    print("Test CLI missing arguments tests passed.")

