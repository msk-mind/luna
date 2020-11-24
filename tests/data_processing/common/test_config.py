#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''
import pytest

from data_processing.common.config import ConfigSet


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


def test_singleton_with_schema():
    c1 = ConfigSet(name='app_config', config_file='tests/data_processing/common/test_config.yml')
    c2 = ConfigSet(name='data_config', config_file='tests/data_processing/common/test_data_ingestion_template.yml',
                   schema_file='data_ingestion_template_schema.yml')

    assert c1 == c2

    assert c1.get_names() == ['app_config', 'data_config']
    assert c2.get_names() == ['app_config', 'data_config']

