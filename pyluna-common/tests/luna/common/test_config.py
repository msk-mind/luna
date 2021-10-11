#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''
import pytest

from luna.common.config import ConfigSet
from luna.common.utils import get_absolute_path


@pytest.fixture(autouse=True)
def cfg():
    print('------setup------')
    c1 = ConfigSet()
    if c1 is not None:
        c1.clear()

    yield cfg

    print('------teardown------')
    c1 = ConfigSet()
    if c1 is not None:
        c1.clear()


def test_singleton_invocations(cfg):
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')
    c2 = ConfigSet()

    assert c1 == c2  # instance is reused for the same config.yaml

    assert c1.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 111


def test_singleton(cfg):
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')
    c2 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')

    assert c1 == c2  # instance is reused for the same config.yaml

    assert c1.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 111

    c3 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/another_test_config.yml')
    c4 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/another_test_config.yml')

    # instance is reused for different configs too
    assert c3 == c4
    assert c2 == c3

    # but values are reloaded when different config file is provided for the same logical name
    assert c3.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 222
    assert c4.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 222


def test_parse_path():
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')

    parsed = c1._parse_path('name::jsonpath')

    assert parsed['name'] == 'name'
    assert parsed['jsonpath'] == 'jsonpath'

    parsed = c1._parse_path('name::jsonpath::jsonpath')

    assert parsed['name'] == 'name'
    assert parsed['jsonpath'] == 'jsonpath::jsonpath'

    with pytest.raises(ValueError) as ve:
        c1._parse_path('name:jsonpath')

    assert "Illegal config path: " in str(ve.value)



def test_get_value(cfg):
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')
    c3 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')

    assert c1.get_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]') == 111

    with pytest.raises(ValueError):
        c3.get_value(path='app_config::$.spark_application_config[:1]["doesnt.exist"]') == None

    with pytest.raises(ValueError):
        ConfigSet().get_value('name_does_not_exist::$.spark_application_config[:1]["spark.executor.cores"]')


def test_get_keys(cfg):
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')

    assert c1.get_keys('app_config') == ['spark_cluster_config', 'spark_application_config']

    with pytest.raises(ValueError):
        ConfigSet().get_keys('name_does_not_exist')


def test_singleton_with_schema(cfg):
    schema_file = get_absolute_path(__file__, '../../../../pyluna-core/luna/data_ingestion_template_schema.yml')

    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')
    c2 = ConfigSet(name='data_config', config_file='pyluna-common/tests/luna/common/test_data_ingestion_template.yml',
                   schema_file=schema_file)

    assert c1 == c2

    assert c1.get_names() == ['app_config', 'data_config']
    assert c2.get_names() == ['app_config', 'data_config']


def test_invalid_yaml(cfg):
    with pytest.raises(IOError):
        ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/does_not_exist.yml')


def test_has_value(cfg):
    c1 = ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')

    assert c1.has_value(path='app_config::$.spark_application_config[:1]["spark.executor.cores"]')
    assert not c1.has_value(path='app_config::$.spark_application_config[:1]["spark.does.not.exist"]')

    with pytest.raises(ValueError):
        ConfigSet().has_value('name_does_not_exist::$.spark_application_config[:1]["spark.does.not.exist"]')

def test_clear(cfg):
    schema_file = get_absolute_path(__file__, '../../../../pyluna-core/luna/data_ingestion_template_schema.yml')

    ConfigSet(name='app_config', config_file='pyluna-common/tests/luna/common/test_config.yml')
    ConfigSet(name='data_config', config_file='pyluna-common/tests/luna/common/test_data_ingestion_template.yml',
                   schema_file=schema_file)
    c1 = ConfigSet()
    assert c1.get_names() == ['app_config', 'data_config']
    c1.clear()
    assert c1.get_names() == []
