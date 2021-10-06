import pytest
import os, shutil
import click

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig

from luna.get_pathology_annotations import get_pathology_annotations
from luna.common.DataStore import DataStore_v2

from pytest_mock import mocker
from mock import patch
import json


@pytest.fixture
def client():

    APP_CFG = "test_get_path_annots"
    cfg = ConfigSet(name=APP_CFG, config_file='tests/src/get_pathology_annotations/test_get_pathology_annotations_config.yaml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-get-pathology-annotations-api')


    get_pathology_annotations.app.config['TESTING'] = True
    get_pathology_annotations.app.config['pathology_root_path'] = cfg.get_value(path=APP_CFG+'::$.pathology[:1]["root_path"]')
    get_pathology_annotations.app.config['spark'] = spark

    with get_pathology_annotations.app.test_client() as client:
        yield client


def test_get_point_annotation(mocker, client, monkeypatch):

    response = client.get('/mind/api/v1/getPathologyAnnotation/test-project/123456/point/LYMPHOCYTE_DETECTION_LABELSET')
    assert b"[{\"type\": \"Feature\", \"id\": \"PathAnnotationObject" in response.data


def test_get_regional_annotation(mocker, client, monkeypatch):

    def mock_datastore_get(*args, **kwargs):
        if kwargs['store_id'] == '123456':
            return 'pyluna-pathology/tests/luna/pathology/common/testdata/project/slides/123/CONCAT/RegionalAnnotationJSON/DEFAULT'
        else:
            raise RuntimeWarning(f"Data not found at 123456")

    monkeypatch.setattr(DataStore_v2, "get", mock_datastore_get)
        
    response = client.get('/mind/api/v1/getPathologyAnnotation/test-project/123456/regional/DEFAULT_LABELS')

    response_data = json.loads(response.get_data(as_text=True))
    assert "features" in response_data



def test_get_bad_slide_id(mocker, client, monkeypatch):
  
    def mock_datastore_get(*args, **kwargs):
        if kwargs['store_id'] == '123456':
            return 'tests/src/pathology/common/testdata/project/slides/123/CONCAT/RegionalAnnotationJSON/DEFAULT'
        else:
            raise RuntimeWarning(f"Data not found at 123456")

    monkeypatch.setattr(DataStore_v2, "get", mock_datastore_get)

    response = client.get('/mind/api/v1/getPathologyAnnotation/test-project/1/regional/DEFAULT_LABELS')
    print(response)
    print(response.data)

    assert response.data==b'Invalid ID'


# @patch.dict(PROJECT_MAPPING, {'test': 'test-project'}, clear=True)
# def test_get_no_match(mocker, client):

#     response = client.get('/mind/api/v1/getPathologyAnnotation/test/1234562/regional/DEFAULT_LABELS')
#     assert response.data == b"No annotations match the provided query."
