import os, pytest
from pathlib import Path
from datetime import datetime

from luna.pathology.common.annotation_utils import *

SLIDEVIEWER_API_URL = "http://test-slideviewer-url.com/"
SLIDE_BMP_DIR = "pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_bmps"
SLIDE_NPY_DIR = "pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_npys"
TMP_ZIP_DIR = "pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_tmp"
BMP_FILE_PATH= "pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_bmps/1_123/123_user_SVBMP-01c9ea8d50971412600b83d4918d3888818b77792f4c40a66861bbd586ae5d51_annot.bmp"
NPY_FILE_PATH = "pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_npys/1_123/123_user_SVBMP-01c9ea8d50971412600b83d4918d3888818b77792f4c40a66861bbd586ae5d51_annot.npy"
SLIDE_STORE_DIR = "pyluna-pathology/tests/luna/pathology/common/testdata/project/slides"


@pytest.fixture(scope='session', autouse=True)
def clean_after_all_Tests():
    yield
    # Will be executed after the last test
    shutil.rmtree(SLIDE_BMP_DIR)
    shutil.rmtree(SLIDE_NPY_DIR)


def test_get_slide_bitmap(requests_mock):
    requests_mock.get("http://test-slideviewer-url.com/slides/user@mskcc.org/projects;1;1;123.svs/getLabelFileBMP",
                      content=Path('pyluna-pathology/tests/luna/pathology/common/testdata/input/label.zip').read_bytes())

    res = get_slide_bitmap("1;123.svs", "user", "123", SLIDE_BMP_DIR, SLIDEVIEWER_API_URL, TMP_ZIP_DIR, '1')

    assert os.path.exists(res[1])
    assert BMP_FILE_PATH == res[1]


def test_convert_bmp_to_npy():
    actual_path = convert_bmp_to_npy(BMP_FILE_PATH, SLIDE_NPY_DIR)

    assert actual_path == NPY_FILE_PATH


def test_check_slideviewer_and_download_bmp(requests_mock):
    requests_mock.get("http://test-slideviewer-url.com/slides/user@mskcc.org/projects;1;1;123.svs/getLabelFileBMP",
                      content=Path('pyluna-pathology/tests/luna/pathology/common/testdata/input/label.zip').read_bytes())

    res = check_slideviewer_and_download_bmp('1', '1;123.svs', '123', ['user'], SLIDE_BMP_DIR,
                                           SLIDEVIEWER_API_URL, TMP_ZIP_DIR)
    assert len(res) == 2
    assert res[1]['bmp_filepath'] == BMP_FILE_PATH


def test_convert_slide_bitmap_to_geojson():
    outputs = [
        {'sv_project_id': '1', 'slideviewer_path': '1;123.svs', 'slide_id': '123', 'user': 'n/a',
         'bmp_filepath': 'n/a', 'npy_filepath': 'n/a', 'geojson': 'n/a', 'geojson_path': 'n/a', 'date': datetime(2021, 7, 28, 10, 23, 27, 816088)},
        {'sv_project_id': '1', 'slideviewer_path': '1;123.svs', 'slide_id': '123', 'user': 'user',
         'bmp_filepath': 'pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_bmps/1_123/123_user_SVBMP-01c9ea8d50971412600b83d4918d3888818b77792f4c40a66861bbd586ae5d51_annot.bmp',
         'npy_filepath': 'pyluna-pathology/tests/luna/pathology/common/testdata/project/regional_npys/1_123/123_user_SVBMP-01c9ea8d50971412600b83d4918d3888818b77792f4c40a66861bbd586ae5d51_annot.npy',
         'geojson': 'n/a', 'geojson_path': 'n/a', 'date': datetime(2021, 7, 28, 10, 23, 27, 816088)}]

    labelset = {"DEFAULT_LABELS": {1:'tumor', 2:'stroma', 3:'lymphocytes', 4:'adipocytes'}}
    res = convert_slide_bitmap_to_geojson(outputs, labelset, 0.5, SLIDE_NPY_DIR, SLIDE_STORE_DIR)

    assert res[0] == '123'
    assert isinstance(res[1][0]['geojson'], str)
