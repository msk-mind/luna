import pytest
import os, shutil, sys
import requests
from click.testing import CliRunner

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const
from data_processing.pathology.point_annotation.proxy_table import generate
from data_processing.pathology.point_annotation.proxy_table.generate import create_proxy_table, download_point_annotation, cli
from tests.data_processing.pathology.common.request_mock import PointJsonResponse

project_path = "tests/data_processing/pathology/point_annotation/testdata/test-project"
point_json_table_path = project_path + "/tables/POINT_RAW_JSON_ds"
SLIDEVIEWER_URL = None
ROOT_PATH = None
PROJECT = None
PROJECT_PATH = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    cfg = ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    cfg = ConfigSet(name=const.DATA_CFG,
              config_file='tests/data_processing/pathology/point_annotation/testdata/point_js_config.yaml')
    module.spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')

    module.SLIDEVIEWER_URL = cfg.get_value(path=const.DATA_CFG + '::SLIDEVIEWER_URL')
    module.ROOT_PATH = cfg.get_value(path=const.DATA_CFG + '::ROOT_PATH')
    module.PROJECT = cfg.get_value(path=const.DATA_CFG + '::PROJECT')

    module.PROJECT_PATH = f"{ROOT_PATH}/{PROJECT}/configs/POINT_RAW_JSON_ds"
    if os.path.exists(module.PROJECT_PATH):
        shutil.rmtree(module.PROJECT_PATH)
    os.makedirs(module.PROJECT_PATH)

def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    if os.path.exists(point_json_table_path):
        shutil.rmtree(point_json_table_path)
    if os.path.exists(PROJECT_PATH):
        shutil.rmtree(PROJECT_PATH)

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

