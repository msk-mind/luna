#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''


def test_get_value():
    from data_processing.common.config import Config

    c1 = Config('tests/data_processing/common/test_config.yaml')
    c2 = Config('tests/data_processing/common/test_config.yaml')
    c3 = Config('tests/data_processing/common/test_config.yaml')

    print(str(c1) + ' ' + str(c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c2) + ' ' + str(c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c3) + ' ' + str(c1.get_value('$.spark_application_config[:1]["doesnt_exist"]')))

    assert c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c2.get_value('$.spark_application_config[:1]["spark.executor.cores"]') == 111
    assert c3.get_value('$.spark_application_config[:1]["doesnt.exist"]') == None



