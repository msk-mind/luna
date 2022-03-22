# -*- coding: utf-8 -*-
"""
Created on October 17, 2019

@author: aukermaa@mskcc.org
"""
from luna.common.config import ConfigSet
import luna.common.constants as const


def test_table_name():
    c1 = ConfigSet(
        name=const.DATA_CFG,
        config_file="pyluna-common/tests/luna/common/testdata/data_ingestion_template_valid.yml",
    )

    assert const.TABLE_NAME(c1) == "CT_OV_16-158_CT_20201028"


def test_table_location():
    c1 = ConfigSet(
        name=const.DATA_CFG,
        config_file="pyluna-common/tests/luna/common/testdata/data_ingestion_template_valid.yml",
    )

    assert (
        const.TABLE_LOCATION(c1)
        == "pyluna-radiology/tests/luna/radiology/proxy_table/test_data/OV_16-158/tables/CT_OV_16-158_CT_20201028"
    )


def test_table_location_emptystring():
    c1 = ConfigSet(
        name=const.DATA_CFG,
        config_file="pyluna-common/tests/luna/common/testdata/data_ingestion_template_valid_empty_dataset.yml",
    )

    assert (
        const.TABLE_LOCATION(c1)
        == "pyluna-common/tests/luna/radiology/proxy_table/test_data/OV_16-158/tables/CT"
    )


def test_table_location_none():
    c1 = ConfigSet(
        name=const.DATA_CFG,
        config_file="pyluna-common/tests/luna/common/testdata/data_ingestion_template_valid_empty_dataset_2.yml",
    )

    assert (
        const.TABLE_LOCATION(c1)
        == "pyluna-common/tests/luna/radiology/proxy_table/test_data/OV_16-158/tables/CT"
    )


def test_project_location():
    c1 = ConfigSet(
        name=const.DATA_CFG,
        config_file="pyluna-common/tests/luna/common/testdata/data_ingestion_template_valid.yml",
    )

    assert (
        const.PROJECT_LOCATION(c1)
        == "pyluna-radiology/tests/luna/radiology/proxy_table/test_data/OV_16-158"
    )
