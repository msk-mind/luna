'''
Created on October 30, 2020

@author: pashaa@mskcc.org
'''
import os

import pytest
from yamale import YamaleTestCase, YamaleError

from luna.common.utils import get_absolute_path


class TestValidYaml(YamaleTestCase):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    schema = get_absolute_path(__file__, '../../luna/data_ingestion_template_schema.yml')
    yaml = 'data_ingestion_template_valid.yml'

    def runTest(self):
        self.assertTrue(self.validate())


class TestInvalidYaml(YamaleTestCase):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    schema = get_absolute_path(__file__, '../../luna/data_ingestion_template_schema.yml')
    yaml = 'data_ingestion_template_invalid.yml'

    def runTest(self):
        with pytest.raises(YamaleError) as error_info:
            self.assertFalse(self.validate())

        assert "REQUESTOR: 'None' is not a str" in str(error_info.value)
        assert "REQUESTOR_DEPARTMENT: 'None' is not a str" in str(error_info.value)
        assert "REQUESTOR_EMAIL: 'None' is not a str" in str(error_info.value)
        assert "PROJECT: 'None' is not a str" in str(error_info.value)
        assert "SOURCE: 'None' is not a str" in str(error_info.value)
        assert "MODALITY: 'None' not in" in str(error_info.value)
        assert "DATA_TYPE: 'CAT' not in" in \
               str(error_info.value)
        assert "DATE: '555-3' is not a day" in str(error_info.value)
        assert "BWLIMIT: '5T' is not a regex match." in str(error_info.value)
