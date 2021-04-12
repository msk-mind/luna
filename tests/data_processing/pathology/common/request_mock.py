'''
Created on February 01, 2021

@author: pashaa@mskcc.org
'''
from pathlib import Path

class CSVMockResponse:

    content = b'Title: IRB #16-1144 Subset\n' \
                    b'Description: Subset of cases from related master project #141\n' \
                    b'Users: jane@mskcc.org, jo@mskcc.org\n' \
                    b'CoPathTest: false\n' \
                    b'2013;HobS13-283072057510;1435197.svs\n' \
                    b'2013;HobS13-283072057511;1435198.svs\n' \
                    b'2013;HobS13-283072057512;1435199.svs\n'


class ZIPMockResponse:

    def iter_content(self, chunk_size=128):
        return [Path('tests/data_processing/pathology/proxy_table/'
                            'regional_annotation/test_data/input/CMU-1.zip').read_bytes()]


class PointJsonResponse:

    content = b'[{"project_id":"8","image_id":"123.svs",' \
              b'"label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},' \
              b'{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774",' \
              b'"class":"3","classname":"Tissue 4"}]'

    def json(self):
        return [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},
                {"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json_data

    def json(self):
        return eval(self.json_data)
