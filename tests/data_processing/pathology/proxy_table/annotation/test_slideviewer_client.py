'''
Created on January 31, 2021

@author: pashaa@mskcc.org
'''
import os

import pytest
from pytest_mock import mocker
import requests
from data_processing.common.config import ConfigSet
from data_processing.common.constants import DATA_CFG
from data_processing.pathology.proxy_table.annotation.slideviewer_client import get_slide_id, fetch_slide_ids


SLIDEVIEWER_CSV_FILE = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    ConfigSet(
        name=DATA_CFG,
        config_file='tests/data_processing/pathology/proxy_table/annotation/data_config.yaml')
    cfg = ConfigSet()
    LANDING_PATH = cfg.get_value(path=DATA_CFG + '::LANDING_PATH')
    PROJECT_ID = cfg.get_value(path=DATA_CFG + '::PROJECT_ID')
    module.SLIDEVIEWER_CSV_FILE = os.path.join(LANDING_PATH, 'project_' + str(PROJECT_ID) + '.csv')


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    if os.path.exists(module.SLIDEVIEWER_CSV_FILE):
        os.remove(module.SLIDEVIEWER_CSV_FILE)


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

    slides = fetch_slide_ids()

    assert slides == [['2013;HobS13-283072057510;145197.svs', '145197'],
                      ['2013;HobS13-283072057511;145198.svs', '145198'],
                      ['2013;HobS13-283072057512;145199.svs', '145199']]


class MockResponse:

    content = b'Title: IRB #16-1144 Subset\n' \
                    b'Description: Subset of cases from related master project #141\n' \
                    b'Users: jane@mskcc.org, jo@mskcc.org\n' \
                    b'CoPathTest: false\n' \
                    b'2013;HobS13-283072057510;1435197.svs\n' \
                    b'2013;HobS13-283072057511;1435198.svs\n' \
                    b'2013;HobS13-283072057512;1435199.svs\n'

# when optional field SLIDEVIEWER_CSV_FILE is not specified in the data config yaml
def test_fetch_slide_ids_without_csv(monkeypatch):
    # works for any url argument
    def mock_get(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    slides = fetch_slide_ids()

    assert slides == [['2013;HobS13-283072057510;1435197.svs', '1435197'],
                      ['2013;HobS13-283072057511;1435198.svs', '1435198'],
                      ['2013;HobS13-283072057512;1435199.svs', '1435199']]

    assert os.path.isfile(SLIDEVIEWER_CSV_FILE) == True


