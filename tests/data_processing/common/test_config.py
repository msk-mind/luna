#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''
from data_processing.common.config import Config

def test_singleton():
    c1 = Config('tests/data_processing/common/test_config.yaml')
    c2 = Config('tests/data_processing/common/test_config.yaml')

    assert c1 == c2  # instance is reused for the same config.yaml

    assert c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 111

    c3 = Config('tests/data_processing/common/another_test_config.yaml')
    c4 = Config('tests/data_processing/common/another_test_config.yaml')

    assert c3 == c4 # but new instance is created for a new config yaml
    assert c2 != c3

    assert c3.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 222
    assert c4.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 222


def test_get_value():
    c1 = Config('tests/data_processing/common/test_config.yaml')
    c3 = Config('tests/data_processing/common/test_config.yaml')

    assert c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c3.get_value('$.spark_application_config[:1]["doesnt.exist"]') == None



