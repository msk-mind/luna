import pytest
import os, shutil, sys
import requests

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.pathology.point_annotation.proxy_table import generate
from data_processing.pathology.point_annotation.proxy_table.generate import create_proxy_table, download_point_annotation
from tests.data_processing.pathology.common.request_mock import PointJsonResponse

point_json_table_path = "tests/data_processing/pathology/point_annotation/testdata/test-project/tables/POINT_RAW_JSON_ds"
SLIDEVIEWER_URL = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    cfg = ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    cfg = ConfigSet(name=const.DATA_CFG,
              config_file='tests/data_processing/pathology/point_annotation/testdata/point_js_config.yaml')
    module.spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')

    module.SLIDEVIEWER_URL = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_URL')

def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    if os.path.exists(point_json_table_path):
        shutil.rmtree(point_json_table_path)

def test_download_point_annotation(monkeypatch):

    # works for any url argument
    def mock_get(*args, **kwargs):
        return PointJsonResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    import data_processing
    sys.modules['slideviewer_client'] = data_processing.pathology.common.slideviewer_client

    res = download_point_annotation('http://test', "123.svs", 8, "username")

    assert res == [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},
                   {"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]


def test_create_proxy_table(monkeypatch):

    def mock_download(*args, **kwargs):
        return [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},
                {"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]

    monkeypatch.setattr(generate, "download_point_annotation", mock_download)

    create_proxy_table()

    df = spark.read.format("delta").load(point_json_table_path)

    assert 2 == df.count()
    assert set(["slideviewer_path","slide_id","sv_project_id","user",
                "sv_json", "sv_json_record_uuid", "latest", "date_added", "date_updated"]) == set(df.columns)

    df.unpersist()

