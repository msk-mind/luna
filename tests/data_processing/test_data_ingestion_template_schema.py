'''
Created on October 30, 2020

@author: pashaa@mskcc.org
'''
import os

import pytest
from yamale import YamaleTestCase, YamaleError


class TestValidYaml(YamaleTestCase):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    schema = '../../data_ingestion_template_schema.yaml'
    yaml = 'data_ingestion_template_valid.yaml'

    def runTest(self):
        self.assertTrue(self.validate())


class TestInvalidYaml(YamaleTestCase):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    schema = '../../data_ingestion_template_schema.yaml'
    yaml = 'data_ingestion_template_invalid.yaml'

    def runTest(self):
        with pytest.raises(YamaleError) as error_info:
            self.assertFalse(self.validate())

        assert "requestor: 'None' is not a str" in str(error_info.value)
        assert "requestor_department: 'None' is not a str" in str(error_info.value)
        assert "requestor_email: 'None' is not a str" in str(error_info.value)
        assert "project: 'None' is not a str" in str(error_info.value)
        assert "source: 'None' is not a str" in str(error_info.value)
        assert "modality: 'None' not in ('clinical', 'radiology', 'pathology', 'genomics')" in str(error_info.value)
        assert "data_type: 'CAT' not in ('diagnosis', 'medication', 'treatment', 'CT', 'MRI', 'PET')" in \
               str(error_info.value)
        assert "date: '2020-10-29' is not a timestamp" in str(error_info.value)
        assert "bwlimit: '5T' is not a regex match." in str(error_info.value)