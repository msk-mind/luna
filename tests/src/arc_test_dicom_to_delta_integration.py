#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nifi` package.

To run

cd nifi
pytest -s test/test_dicom_to_delta_integration.py
"""

import pytest
from click.testing import CliRunner

import os, sys, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../luna/')))
from src.dicom_to_delta import *
from luna.common.sparksession import SparkConfig
from pyspark.sql import SparkSession

HDFS = "file://"
DATASET_ADDRESS = os.path.join(os.getcwd(), "tests/luna/testdata/test_dataset_address")
DELTA_TABLE_PATH = os.path.join(DATASET_ADDRESS, "table")

BINARY_TABLE = os.path.join(DELTA_TABLE_PATH, "dicom_binary")
DCM_TABLE = os.path.join(DELTA_TABLE_PATH, "dicom")
TABLES = [BINARY_TABLE, DCM_TABLE]


@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    APP_CFG = 'APP_CFG'
    ConfigSet(name=APP_CFG, config_file='tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-dicom-to-delta')

    yield spark

    print('------teardown------')
    for table_path in TABLES:
        if os.path.exists(table_path):
            shutil.rmtree(table_path)

def assertions(spark):
    for table in TABLES:
        df = spark.read.format("delta").load(table)
        assert 3 == df.count()
    df.unpersist()

def test_write_to_delta(spark):

	write_to_delta(spark, HDFS, DATASET_ADDRESS, False, False)
	assertions(spark)



def test_write_to_delta_merge(spark):

	write_to_delta(spark, HDFS, DATASET_ADDRESS, True, False)
	assertions(spark)
