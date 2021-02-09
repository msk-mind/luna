import pytest
import os, shutil
import click

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
import data_processing.common.constants as const

from data_processing.get_pathology_annotations import get_pathology_annotations
from data_processing.get_pathology_annotations.get_pathology_annotations import PROJECT_MAPPING

from pytest_mock import mocker
from mock import patch


@pytest.fixture
def client():

    APP_CFG = "test_get_path_annots"
    cfg = ConfigSet(name=APP_CFG, config_file='tests/data_processing/get_pathology_annotations/test_get_pathology_annotations_config.yaml')
    spark = SparkConfig().spark_session(config_name=APP_CFG, app_name='test-get-pathology-annotations-api')


    get_pathology_annotations.app.config['TESTING'] = True
    get_pathology_annotations.app.config['pathology_root_path'] = cfg.get_value(path=APP_CFG+'::$.pathology[:1]["root_path"]')
    get_pathology_annotations.app.config['spark'] = spark

    with get_pathology_annotations.app.test_client() as client:
        yield client


@patch.dict(PROJECT_MAPPING, {'test': 'test-project'}, clear=True)
def test_get_point_annotation(mocker, client):

    response = client.get('/mind/api/v1/getPathologyAnnotation/test/123456/point/LYMPHOCYTE_DETECTION_LABELSET')
    assert b"[{\"type\":\"Feature\",\"id\":\"PathAnnotationObject" in response.data


@patch.dict(PROJECT_MAPPING, {'test': 'test-project'}, clear=True)
def test_get_regional_annotation(mocker, client):

    response = client.get('/mind/api/v1/getPathologyAnnotation/test/123456/regional/DEFAULT_LABELS')
    assert b"{\"type\":\"FeatureCollection\",\"features" in response.data


@patch.dict(PROJECT_MAPPING, {'test': 'test-project'}, clear=True)
def test_get_bad_slide_id(mocker, client):

    response = client.get('/mind/api/v1/getPathologyAnnotation/test/1/regional/DEFAULT_LABELS')
    assert response.data == b"Invalid ID"


@patch.dict(PROJECT_MAPPING, {'test': 'test-project'}, clear=True)
def test_get_no_match(mocker, client):

    response = client.get('/mind/api/v1/getPathologyAnnotation/test/1234562/regional/DEFAULT_LABELS')
    assert response.data == b"No annotations match the provided query."