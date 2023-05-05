import os

import fire
import pandas as pd
import pytest

from luna.pathology.cli.slide_etl import cli


@pytest.fixture(autouse=True)
def env(monkeypatch):
    print("------setup------")
    # setup env
    monkeypatch.setenv("HOSTNAME", "localhost")


def test_slide_etl(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide_urlpath",
            "tests/testdata/pathology/test-project/wsi",
            "--output-urlpath",
            str(tmp_path),
            "--project_name",
            "TEST-00-000",
            "--comment",
            "Test ingestion",
        ],
    )

    df = pd.read_parquet(os.path.join(tmp_path, "slide_ingest_TEST-00-000.parquet"))

    df = df.set_index("id")
    print(df)

    assert df.loc["123", "slide_size"] == 1938955
    assert df.loc["123", "properties.aperio.AppMag"] == 20
    assert df.loc["123", "url"] == "file://" + os.path.join(tmp_path, "123.svs")

    assert os.path.exists(os.path.join(tmp_path, "123.svs"))

    assert os.path.exists(os.path.join(tmp_path, "fake_slide.svs"))


def test_slide_etl_subset(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/test-project/wsi",
            "--output-urlpath",
            str(tmp_path),
            "--project_name",
            "TEST-00-000",
            "--comment",
            "Test ingestion",
            "--subset-csv",
            "tests/testdata/pathology/slide_etl_subset.csv",
        ],
    )

    df = pd.read_parquet(os.path.join(tmp_path, "slide_ingest_TEST-00-000.parquet"))

    df = df.set_index("id")
    print(df)

    assert df.loc["123", "slide_size"] == 1938955
    assert df.loc["123", "properties.aperio.AppMag"] == 20
    assert df.loc["123", "url"] == "file://" + os.path.join(tmp_path, "123.svs")

    assert os.path.exists(os.path.join(tmp_path, "123.svs"))

    assert len(df) == 1
