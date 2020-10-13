import os, shutil
import pytest
from pytest_mock import mocker
from click.testing import CliRunner
from data_processing.preprocess_feature import cli, generate_feature_table, lookup_dmp_patient_id
from data_processing.common.sparksession import SparkConfig

BASE_DIR = "./tests/data_processing/testdata/"
DESTINATION_DIR = "./tests/data_processing/testdata/outputdir"
TARGET_SPACING = (1.0, 1.0, 3.0)

# Test CLI parameters
runner = CliRunner()

@pytest.fixture(autouse=True)
def spark(monkeypatch):
    print('------setup------')
    monkeypatch.setenv("GRAPH_URI", "bolt://localhost:7883")

    spark = SparkConfig().spark_session('test-preprocessing-feature', 'local[2]')
    yield spark

    print('------teardown------')
    feature_table = os.path.join(BASE_DIR, "data/radiology/tables/radiology.feature-table-test-name")
    if os.path.exists(feature_table):
        shutil.rmtree(feature_table)

    feature_dir = os.path.join(BASE_DIR, "data/radiology/features")
    if os.path.exists(feature_dir):
        shutil.rmtree(feature_dir)

    if os.path.exists(DESTINATION_DIR):
        shutil.rmtree(DESTINATION_DIR)

def test_local_feature_table_generation(mocker, spark):
    # mock graph connection helper method
    mocker.patch('data_processing.preprocess_feature.lookup_dmp_patient_id', side_effect=['P-0019027'])

    # Test no query, default naming
    # Build Feature Table
    generate_feature_table(BASE_DIR, BASE_DIR, TARGET_SPACING, spark, "SeriesInstanceUID = '1.2.840.113619.2.55.3.2743925538.934.1319713655.582'", "feature-table-test-name", "tests/external_process_patient_script.py")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "data/radiology/tables/radiology.feature-table-test-name")
    
    # Read Delta Table and Verify 
    feature_df = spark.read.format("delta").load(feature_table_path)
    assert feature_df.count() == 1
    print ("test_local_feature_table_generation passed.")


def test_local_feature_table_generation_malformed_query(spark):

    # Test no query, default naming
    # Build Feature Table
    generate_feature_table(BASE_DIR, BASE_DIR, TARGET_SPACING, spark, "SeriesInstanceUID = '123' || scan_record_uuid = '123'", "feature-table", "tests/test_external_process_patient_script.py")

    # read and verify correct feature table generated
    feature_table_path = os.path.join(BASE_DIR, "data/radiology/tables/radiology.feature-table")
    
    assert not os.path.exists(feature_table_path)
    print ("test_local_feature_table_generation_malformed_query passed.")


def test_feature_table_cli():
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {} --target_spacing 1 1 3 --query \"SeriesInstanceUID = \'1.2.840.113619.2.55.3.2743925538.934.1319713655.579\'\" --feature_table_output_name test-cli-2 --custom_preprocessing_script tests/test_external_process_patient_script.py".format(BASE_DIR))
    assert result.exit_code == 0
    print("CLI test passed.")

def test_feature_table_cli_with_destination_directory():
    assert os.path.exists(DESTINATION_DIR) == False
    result = runner.invoke(cli, "--spark_master_uri local[*] --base_directory {0} --destination_directory {1} --target_spacing 1 1 3 --query \"SeriesInstanceUID = \'1.2.840.113619.2.55.3.2743925538.934.1319713655.579\'\" --feature_table_output_name test-cli-2 --custom_preprocessing_script tests/test_external_process_patient_script.py".format(BASE_DIR, DESTINATION_DIR))
    assert result.exit_code == 0
    assert os.path.exists(DESTINATION_DIR) == True
    print("CLI test passed.")


def test_feature_table_cli_missing_params():
    test_args = ["--base_directory {} --target_spacing 1 1 3".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {}".format(BASE_DIR),
            "--spark_master_uri local[*] --base_directory {} --target_spacing 1.0 1.0 3.0".format("/path/does/not/exist")]

    for arg in test_args:
        result = runner.invoke(cli, arg)
        assert result.exit_code == 2

    print("Test CLI missing arguments tests passed.")

