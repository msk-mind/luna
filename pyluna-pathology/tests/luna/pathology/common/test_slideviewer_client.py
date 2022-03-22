"""
Created on January 31, 2021

@author: pashaa@mskcc.org
"""
import os, sys
import shutil
import requests
from pathlib import Path
from luna.common.config import ConfigSet
from luna.common.constants import DATA_CFG, CONFIG_LOCATION, PROJECT_LOCATION
from luna.pathology.common.slideviewer_client import (
    get_slide_id,
    fetch_slide_ids,
    download_zip,
    unzip,
    download_sv_point_annotation,
)

SLIDEVIEWER_API_URL = None
SLIDEVIEWER_CSV_FILE = None
LANDING_PATH = None
PROJECT_ID = None
PROJECT = None
zipfile_path = None
PROJECT_PATH = None
ROOT_PATH = None


def setup_module(module):
    """setup any state specific to the execution of the given module."""
    ConfigSet(
        name=DATA_CFG,
        config_file="pyluna-pathology/tests/luna/pathology/common/testdata/data_config_with_slideviewer_csv.yaml",
    )
    cfg = ConfigSet()
    module.SLIDEVIEWER_API_URL = cfg.get_value(path=DATA_CFG + "::SLIDEVIEWER_API_URL")
    module.LANDING_PATH = cfg.get_value(path=DATA_CFG + "::LANDING_PATH")
    module.PROJECT_ID = cfg.get_value(path=DATA_CFG + "::PROJECT_ID")
    module.PROJECT = cfg.get_value(path=DATA_CFG + "::PROJECT")
    module.ROOT_PATH = cfg.get_value(path=DATA_CFG + "::ROOT_PATH")
    module.SLIDEVIEWER_CSV_FILE = cfg.get_value(
        path=DATA_CFG + "::SLIDEVIEWER_CSV_FILE"
    )
    module.zipfile_path = os.path.join(LANDING_PATH, "24bpp-topdown-320x240.bmp.zip")

    if os.path.exists(LANDING_PATH):
        shutil.rmtree(LANDING_PATH)
    os.makedirs(LANDING_PATH)

    module.PROJECT_PATH = os.path.join(
        ROOT_PATH, "OV_16-158/configs/H&E_OV_16-158_CT_20201028"
    )
    if os.path.exists(module.PROJECT_PATH):
        shutil.rmtree(module.PROJECT_PATH)
    os.makedirs(module.PROJECT_PATH)


def teardown_module(module):
    """teardown any state that was previously setup with a setup_module
    method.
    """
    shutil.rmtree(LANDING_PATH)


def test_get_slide_id():
    assert "1435197" == get_slide_id("2013;HobS13-283072057510;1435197.svs")


# when optional field SLIDEVIEWER_CSV_FILE is specified in the data config yaml
def test_fetch_slide_ids_with_csv(monkeypatch):
    # pretend like data config has value for SLIDEVIEWER_CSV_FILE
    def mock_get_value(*args, **kwargs):
        if kwargs["path"] == "DATA_CFG::SLIDEVIEWER_CSV_FILE":
            return "pyluna-pathology/tests/luna/pathology/common/testdata/input/slideviewer.csv"
        else:
            return "no_value"

    monkeypatch.setattr(ConfigSet, "get_value", mock_get_value)

    config_dir = f"{ROOT_PATH}/{PROJECT}/configs"

    slides = fetch_slide_ids(None, PROJECT_ID, config_dir, SLIDEVIEWER_CSV_FILE)

    assert os.path.exists(f"{config_dir}/project_{PROJECT_ID}.csv")
    assert slides == [
        ["2013;HobS13-283072057510;145197.svs", "145197", 155],
        ["2013;HobS13-283072057511;145198.svs", "145198", 155],
        ["2013;HobS13-283072057512;145199.svs", "145199", 155],
    ]


# when optional field SLIDEVIEWER_CSV_FILE is not specified in the data config yaml
def test_fetch_slide_ids_without_csv(requests_mock):

    requests_mock.get(
        "https://fake-slides-res.mskcc.org/exportProjectCSV?pid=155",
        content=b"Title: IRB #16-1144 Subset\n"
        b"Description: Subset of cases from related master project #141\n"
        b"Users: jane@mskcc.org, jo@mskcc.org\n"
        b"CoPathTest: false\n"
        b"2013;HobS13-283072057510;1435197.svs\n"
        b"2013;HobS13-283072057511;1435198.svs\n"
        b"2013;HobS13-283072057512;1435199.svs\n",
    )

    config_dir = f"{ROOT_PATH}/{PROJECT}/configs"
    slides = fetch_slide_ids(SLIDEVIEWER_API_URL, PROJECT_ID, config_dir)

    assert os.path.exists(f"{config_dir}/project_{PROJECT_ID}.csv")
    assert slides == [
        ["2013;HobS13-283072057510;1435197.svs", "1435197", 155],
        ["2013;HobS13-283072057511;1435198.svs", "1435198", 155],
        ["2013;HobS13-283072057512;1435199.svs", "1435199", 155],
    ]


def test_downlaod_zip(requests_mock):

    requests_mock.get(
        SLIDEVIEWER_API_URL,
        content=Path(
            "pyluna-pathology/tests/luna/pathology/common/testdata/input/label.zip"
        ).read_bytes(),
    )
    download_zip(SLIDEVIEWER_API_URL, zipfile_path, chunk_size=128)

    assert os.path.isfile(zipfile_path) == True


def test_unzip():
    shutil.copyfile(
        "pyluna-pathology/tests/luna/pathology/common/testdata/input/label.zip",
        zipfile_path,
    )

    unzipped_file_descriptor = unzip(zipfile_path)

    assert len(unzipped_file_descriptor.read("labels.bmp")) == 3299422


def test_download_sv_point_annotation(requests_mock):

    requests_mock.get(
        "http://test/user@8;123.svs/get",
        text='[{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"}, '
        + '{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]',
    )

    import luna.pathology

    sys.modules["slideviewer_client"] = luna.pathology.common.slideviewer_client

    res = download_sv_point_annotation("http://test/user@8;123.svs/get")

    assert res == [
        {
            "project_id": "8",
            "image_id": "123.svs",
            "label_type": "nucleus",
            "x": "1440",
            "y": "747",
            "class": "0",
            "classname": "Tissue 1",
        },
        {
            "project_id": "8",
            "image_id": "123.svs",
            "label_type": "nucleus",
            "x": "1424",
            "y": "774",
            "class": "3",
            "classname": "Tissue 4",
        },
    ]
