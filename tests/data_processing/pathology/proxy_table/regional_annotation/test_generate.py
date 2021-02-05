import datetime

import pandas
import pytest
import requests
from pytest_mock import mocker
import shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os, sys

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.proxy_table.regional_annotation import generate
from data_processing.pathology.proxy_table.regional_annotation.generate import cli, convert_bmp_to_npy, \
    create_proxy_table, process_regional_annotation_slide_row_pandas
import data_processing.common.constants as const
from tests.data_processing.pathology.common.request_mock import CSVMockResponse, \
    ZIPMockResponse


spark = None
LANDING_PATH = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    ConfigSet(name=const.DATA_CFG,
              config_file='tests/data_processing/pathology/common/testdata/data_config_with_slideviewer_csv.yaml')
    module.spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')

    cfg = ConfigSet()
    module.LANDING_PATH = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')

    if os.path.exists(LANDING_PATH):
        shutil.rmtree(LANDING_PATH)
    os.makedirs(LANDING_PATH)


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    shutil.rmtree(LANDING_PATH)


def test_convert_bmp_to_npy():
    actual_path = convert_bmp_to_npy('tests/data_processing/pathology/proxy_table/'
                                         'regional_annotation/test_data/input/labels.bmp',
                                         LANDING_PATH)

    expected_path = os.path.join(LANDING_PATH, 'input/labels.npy')
    assert actual_path == expected_path
    assert os.path.exists(expected_path)


def test_process_regional_annotation_slide_row_pandas(monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    import data_processing
    sys.modules['slideviewer_client'] = data_processing.pathology.common.slideviewer_client

    # mock request to slideviewer api
    def mock_get(*args, **kwargs):
        if 'exportProjectCSV' in args[0]:
            return CSVMockResponse()
        elif 'getLabelFileBMP' in args[0]:
            return ZIPMockResponse()
        else:
            return None

    monkeypatch.setattr(requests, "get", mock_get)

    data = {'slideviewer_path': ['CMU-1.svs'],
            'slide_id': ['CMU-1'],
            'sv_project_id' : ['155'],
            'bmp_filepath': [''],
            'user': ['someuser'],
            'date_added': ['2021-02-02 10:07:55.802143'],
            'date_updated': ['2021-02-02 10:07:55.802143'],
            'bmp_record_uuid': [''],
            'latest': [True],
            'SLIDE_BMP_DIR': ['tests/data_processing/pathology/proxy_table/regional_annotation/test_data/output/regional_bmps'],
            'TMP_ZIP_DIR': ['tests/data_processing/pathology/proxy_table/regional_annotation/test_data/output/gynocology_tmp_zips'],
            'SLIDEVIEWER_API_URL':['https://fakeslides-res.mskcc.org/']}

    df = pandas.DataFrame(data=data)

    df = process_regional_annotation_slide_row_pandas(df)

    assert df['bmp_filepath'].item() == 'tests/data_processing/pathology/proxy_table/regional_annotation' \
                                        '/test_data/output/regional_bmps/CMU-1' \
                                        '/CMU-1_someuser_SVBMP-90649b2e6e64b4925eed1f32bb68560ade249a9c3bf8e9b27bebebe005638375_annot.bmp'
    assert df['bmp_record_uuid'].item() == 'SVBMP-90649b2e6e64b4925eed1f32bb68560ade249a9c3bf8e9b27bebebe005638375'


def test_create_proxy_table(monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    def mock_process(row:  pandas.DataFrame)-> pandas.DataFrame:
        data = {'slideviewer_path': ['CMU-1.svs'],
                'slide_id': ['CMU-1'],
                'sv_project_id': [155],
                'bmp_filepath': ['tests/data_processing/pathology/proxy_table/regional_annotation/test_data/input/labels.bmp'],
                'user': ['someuser'],
                'date_added': [1612403271],
                'date_updated': [1612403271],
                'bmp_record_uuid': ['SVBMP-90836da'],
                'latest': [True],
                'SLIDE_BMP_DIR': [
                    'tests/data_processing/pathology/proxy_table/regional_annotation/test_data/output/regional_bmps'],
                'TMP_ZIP_DIR': [
                    'tests/data_processing/pathology/proxy_table/regional_annotation/test_data/output/gynocology_tmp_zips'],
                'SLIDEVIEWER_API_URL': ['https://fakeslides-res.mskcc.org/']}

        return pandas.DataFrame(data=data)


    monkeypatch.setattr(generate, "process_regional_annotation_slide_row_pandas",
                        mock_process)

    assert create_proxy_table() == 0  # exit code
