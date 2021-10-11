import pandas
import shutil
import os, sys
from pathlib import Path

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.pathology.proxy_table.regional_annotation import generate
from luna.pathology.proxy_table.regional_annotation.generate \
    import create_proxy_table, process_regional_annotation_slide_row_pandas
import luna.common.constants as const

spark = None
LANDING_PATH = None
ROOT_PATH = None
PROJECT_PATH = None

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    ConfigSet(name=const.APP_CFG, config_file='pyluna-radiology/tests/test_config.yml')
    ConfigSet(name=const.DATA_CFG,
              config_file='pyluna-pathology/tests/luna/pathology/common/testdata/data_config_with_slideviewer_csv.yaml')
    module.spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-annotation-proxy')

    cfg = ConfigSet()
    module.LANDING_PATH = cfg.get_value(path=const.DATA_CFG + '::LANDING_PATH')
    module.ROOT_PATH = cfg.get_value(path=const.DATA_CFG + '::ROOT_PATH')

    if os.path.exists(LANDING_PATH):
        shutil.rmtree(LANDING_PATH)
    os.makedirs(LANDING_PATH)
    module.PROJECT_PATH = os.path.join(ROOT_PATH, "OV_16-158/configs/H&E_OV_16-158_CT_20201028")
    if os.path.exists(PROJECT_PATH):
        shutil.rmtree(PROJECT_PATH)
    os.makedirs(PROJECT_PATH)

def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    shutil.rmtree(LANDING_PATH)

def test_process_regional_annotation_slide_row_pandas(monkeypatch, requests_mock):
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    import luna
    sys.modules['slideviewer_client'] = luna.pathology.common.slideviewer_client

    requests_mock.get("https://fakeslides-res.mskcc.org/slides/someuser@mskcc.org/projects;155;CMU-1.svs/getLabelFileBMP",
                      content=Path('pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/input/CMU-1.zip').read_bytes())

    requests_mock.get("https://fake-slides-res.mskcc.org/exportProjectCSV?pid=155",
                      content=b'Title: IRB #16-1144 Subset\n' \
                              b'Description: Subset of cases from related master project #141\n' \
                              b'Users: jane@mskcc.org, jo@mskcc.org\n' \
                              b'CoPathTest: false\n' \
                              b'2013;HobS13-283072057510;1435197.svs\n' \
                              b'2013;HobS13-283072057511;1435198.svs\n' \
                              b'2013;HobS13-283072057512;1435199.svs\n')

    data = {'slideviewer_path': ['CMU-1.svs'],
            'slide_id': ['CMU-1'],
            'sv_project_id' : ['155'],
            'bmp_filepath': [''],
            'user': ['someuser'],
            'date_added': ['2021-02-02 10:07:55.802143'],
            'date_updated': ['2021-02-02 10:07:55.802143'],
            'bmp_record_uuid': [''],
            'latest': [True],
            'SLIDE_BMP_DIR': ['pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/output/regional_bmps'],
            'TMP_ZIP_DIR': ['pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/output/gynocology_tmp_zips'],
            'SLIDEVIEWER_API_URL':['https://fakeslides-res.mskcc.org/']}

    df = pandas.DataFrame(data=data)

    df = process_regional_annotation_slide_row_pandas(df)

    assert df['bmp_filepath'].item() == 'pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation' \
                                        '/test_data/output/regional_bmps/CMU-1' \
                                        '/CMU-1_someuser_SVBMP-90649b2e6e64b4925eed1f32bb68560ade249a9c3bf8e9b27bebebe005638375_annot.bmp'
    assert df['bmp_record_uuid'].item() == 'SVBMP-90649b2e6e64b4925eed1f32bb68560ade249a9c3bf8e9b27bebebe005638375'


def test_create_proxy_table(monkeypatch):
    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    def mock_process(row:  pandas.DataFrame)-> pandas.DataFrame:
        data = {'slideviewer_path': ['CMU-1.svs'],
                'slide_id': ['CMU-1'],
                'sv_project_id': [155],
                'bmp_filepath': ['pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/input/labels.bmp'],
                'user': ['someuser'],
                'date_added': [1612403271],
                'date_updated': [1612403271],
                'bmp_record_uuid': ['SVBMP-90836da'],
                'latest': [True],
                'SLIDE_BMP_DIR': [
                    'pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/output/regional_bmps'],
                'TMP_ZIP_DIR': [
                    'pyluna-pathology/tests/luna/pathology/proxy_table/regional_annotation/test_data/output/gynocology_tmp_zips'],
                'SLIDEVIEWER_API_URL': ['https://fakeslides-res.mskcc.org/']}

        return pandas.DataFrame(data=data)


    monkeypatch.setattr(generate, "process_regional_annotation_slide_row_pandas",
                        mock_process)


    assert create_proxy_table() == 0  # exit code
