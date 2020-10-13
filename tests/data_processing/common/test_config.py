#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''


def test_get_value():
    from data_processing.common.config import Config

    config = Config(config_file='tests/data_processing/common/test_config.yaml')
    assert config.get_value('spark_application_config/spark.executor.cores') == 111
    assert config.get_value('spark_application_config/spark.executor.memory') == '36g'



