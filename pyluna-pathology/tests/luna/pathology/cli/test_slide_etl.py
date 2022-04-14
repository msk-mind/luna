import pytest
import os
from click.testing import CliRunner
from luna.pathology.cli.slide_etl import cli
import pandas as pd


@pytest.fixture(autouse=True)
def env(monkeypatch):
    print("------setup------")
    # setup env
    monkeypatch.setenv("HOSTNAME", "localhost")


def test_slide_etl(tmp_path):
    runner = CliRunner()

    table_location = os.path.join(tmp_path, "tables")
    store_location = os.path.join(tmp_path, "data")

    os.makedirs(store_location)

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/testdata/data/test-project"
            ""
            "/wsi",
            "-o",
            table_location,
            "--store_url",
            f"file://{store_location}",
            "--num_cores",
            1,
            "--project_name",
            "TEST-00-000",
            "--comment",
            "Test ingestion",
        ],
    )

    assert result.exit_code == 0

    df = pd.read_parquet(
        os.path.join(table_location, "slide_ingest_TEST-00-000.parquet")
    )

    print(df)

    assert df.loc["123", "size"] == 1938955
    assert df.loc["123", "aperio.AppMag"] == 20
    assert df.loc["123", "data_url"] == "file://" + os.path.join(
        store_location, "pathology/TEST-00-000/slides/123.svs"
    )

    assert os.path.exists(
        os.path.join(store_location, "pathology/TEST-00-000/slides/123.svs")
    )

    assert (
        len(os.listdir(os.path.join(store_location, "pathology/TEST-00-000/slides/")))
        == 1
    )


def test_slide_etl_subset(tmp_path):
    runner = CliRunner()

    table_location = os.path.join(tmp_path, "tables")
    store_location = os.path.join(tmp_path, "data")

    os.makedirs(store_location)

    result = runner.invoke(
        cli,
        [
            "pyluna-pathology/tests/luna/pathology/testdata/data/test-project"
            ""
            "/wsi",
            "-o",
            table_location,
            "--store_url",
            f"file://{store_location}",
            "--num_cores",
            1,
            "--project_name",
            "TEST-00-000",
            "--comment",
            "Test ingestion",
            "--subset-csv",
            "pyluna-pathology/tests/luna/pathology/cli/testdata"
            "/slide_etl_subset.csv",
        ],
    )

    assert result.exit_code == 0

    df = pd.read_parquet(
        os.path.join(table_location, "slide_ingest_TEST-00-000.parquet")
    )

    print(df)

    assert df.loc["123", "size"] == 1938955
    assert df.loc["123", "aperio.AppMag"] == 20
    assert df.loc["123", "data_url"] == "file://" + os.path.join(
        store_location, "pathology/TEST-00-000/slides/123.svs"
    )

    assert os.path.exists(
        os.path.join(store_location, "pathology/TEST-00-000/slides/123.svs")
    )

    assert (
        len(os.listdir(os.path.join(store_location, "pathology/TEST-00-000/slides/")))
        == 1
    )
