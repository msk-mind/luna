# -*- coding: utf-8 -*-
'''
Created on October 17, 2019

@author: aukermaa@mskcc.org
'''
import pytest

from data_processing.common.config import ConfigSet
import data_processing.common.constants as const

def test_source_table_name():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/pathology/refined_table/regional_annotation/geojson_data.yaml')

    assert const.TABLE_NAME(c1, is_source=True) == "REGIONAL_BITMASK_dsn"

def test_table_name():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/data_ingestion_template_valid.yml')

    assert const.TABLE_NAME(c1) == "CT_OV_16-158_CT_20201028"


def test_table_location():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/data_ingestion_template_valid.yml')

    assert const.TABLE_LOCATION(c1) == "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/tables/CT_OV_16-158_CT_20201028"


def test_table_location_emptystring():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/data_ingestion_template_valid_empty_dataset.yml')
    
    assert const.TABLE_LOCATION(c1) == "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/tables/CT"


def test_table_location_none():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/data_ingestion_template_valid_empty_dataset_2.yml')

    assert const.TABLE_LOCATION(c1) == "tests/data_processing/radiology/proxy_table/test_data/OV_16-158/tables/CT"

def test_project_location():
    c1 = ConfigSet(name=const.DATA_CFG, config_file='tests/data_processing/data_ingestion_template_valid.yml')

    assert const.PROJECT_LOCATION(c1) == "tests/data_processing/radiology/proxy_table/test_data/OV_16-158"

