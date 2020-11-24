#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''
import pytest

from data_processing.common.config import ConfigSet


def test_singleton_invocations():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')
    c2 = ConfigSet(name='app_config')

    assert c1 == c2  # instance is reused for the same config.yaml

    assert c1.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 111


def test_singleton():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')
    c2 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')

    assert c1 == c2  # instance is reused for the same config.yaml

    assert c1.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 111

    c3 = ConfigSet(name='app_config', config_file='tests/data_processing/common/another_test_config.yml')
    c4 = ConfigSet(name='app_config', config_file='tests/data_processing/common/another_test_config.yml')

    # instance is reused for different configs too
    assert c3 == c4
    assert c2 == c3

    # but values are reloaded when different config file is provided for the same logical name
    assert c3.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 222
    assert c4.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 222


def test_get_value():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')
    c3 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')

    assert c1.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]') == 111

    with pytest.raises(ValueError):
        c3.get_value(name='app_config', jsonpath='$.spark_application_config[:1]["doesnt.exist"]') == None


def test_get_keys():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')

    assert c1.get_keys('app_config') == ['spark_cluster_config', 'spark_application_config']


def test_singleton_with_schema():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')
    c2 = ConfigSet(name='data_config', config_file='tests/data_processing/common/test_data_ingestion_template.yml',
                   schema_file='data_ingestion_template_schema.yml')

    assert c1 == c2

    assert c1.get_names() == ['app_config', 'data_config']
    assert c2.get_names() == ['app_config', 'data_config']

def test_invalid_yaml():
    with pytest.raises(IOError):
        c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/does_not_exist.yml')

def test_has_value():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')

    assert c1.has_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.executor.cores"]')
    assert not c1.has_value(name='app_config', jsonpath='$.spark_application_config[:1]["spark.does.not.exist"]')