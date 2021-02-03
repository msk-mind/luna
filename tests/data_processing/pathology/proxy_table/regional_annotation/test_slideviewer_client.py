'''
Created on January 31, 2021

@author: pashaa@mskcc.org
'''
import os
import shutil

import pytest
from pytest_mock import mocker
import requests
from data_processing.common.config import ConfigSet
from data_processing.common.constants import DATA_CFG
from data_processing.pathology.proxy_table.regional_annotation.slideviewer_client import get_slide_id, fetch_slide_ids, \
    download_zip, unzip
from tests.data_processing.pathology.proxy_table.regional_annotation.request_mock import CSVMockResponse, \
    ZIPMockResponse

SLIDEVIEWER_API_URL = None
SLIDEVIEWER_CSV_FILE = None
LANDING_PATH = None
PROJECT_ID = None
zipfile_path = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    ConfigSet(
        name=DATA_CFG,
        config_file='tests/data_processing/pathology/proxy_table/regional_annotation/data_config.yaml')
    cfg = ConfigSet()
    module.SLIDEVIEWER_API_URL = cfg.get_value(path=DATA_CFG + '::SLIDEVIEWER_API_URL')
    module.LANDING_PATH = cfg.get_value(path=DATA_CFG + '::LANDING_PATH')
    module.PROJECT_ID = cfg.get_value(path=DATA_CFG + '::PROJECT_ID')
    module.SLIDEVIEWER_CSV_FILE = cfg.get_value(path=DATA_CFG + '::SLIDEVIEWER_CSV_FILE')
    module.zipfile_path = os.path.join(LANDING_PATH, '24bpp-topdown-320x240.bmp.zip')

    if os.path.exists(LANDING_PATH):
        shutil.rmtree(LANDING_PATH)
    os.makedirs(LANDING_PATH)


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    shutil.rmtree(LANDING_PATH)


def test_get_slide_id():
    assert '1435197' == get_slide_id('2013;HobS13-283072057510;1435197.svs')


# when optional field SLIDEVIEWER_CSV_FILE is specified in the data config yaml
def test_fetch_slide_ids_with_csv(monkeypatch):
    # pretend like data config has value for SLIDEVIEWER_CSV_FILE
    def mock_get_value(*args, **kwargs):
        if kwargs['path'] == DATA_CFG + '::SLIDEVIEWER_CSV_FILE':
            return 'tests/data_processing/pathology/proxy_table/annotation/test_data/input/slideviewer.csv'
        else:
            return 'no_value'

    monkeypatch.setattr(ConfigSet, "get_value", mock_get_value)

    slides = fetch_slide_ids(None, PROJECT_ID, LANDING_PATH, SLIDEVIEWER_CSV_FILE)

    assert slides == [['2013;HobS13-283072057510;145197.svs', '145197'],
                      ['2013;HobS13-283072057511;145198.svs', '145198'],
                      ['2013;HobS13-283072057512;145199.svs', '145199']]


# when optional field SLIDEVIEWER_CSV_FILE is not specified in the data config yaml
def test_fetch_slide_ids_without_csv(monkeypatch):
    # works for any url argument
    def mock_get(*args, **kwargs):
        return CSVMockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    slides = fetch_slide_ids(SLIDEVIEWER_API_URL, PROJECT_ID, LANDING_PATH)

    assert slides == [['2013;HobS13-283072057510;1435197.svs', '1435197'],
                      ['2013;HobS13-283072057511;1435198.svs', '1435198'],
                      ['2013;HobS13-283072057512;1435199.svs', '1435199']]


def test_downlaod_zip(monkeypatch):
    # works for any url argument
    def mock_get(*args, **kwargs):
        return ZIPMockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    download_zip(SLIDEVIEWER_API_URL, zipfile_path, chunk_size=128)


    assert os.path.isfile(zipfile_path) == True

def test_unzip(monkeypatch):
    shutil.copyfile('tests/data_processing/pathology/proxy_table/'
                            'regional_annotation/test_data/input/24bpp-topdown-320x240.bmp.zip',
                    zipfile_path)

    unzipped_file_descriptor = unzip(zipfile_path)

    assert len(unzipped_file_descriptor.read('24bpp-topdown-320x240.bmp')) == 230454
