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
                            'regional_annotation/test_data/input/24bpp-topdown-320x240.bmp.zip').read_bytes()]


