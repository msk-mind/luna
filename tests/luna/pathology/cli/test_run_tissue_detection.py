import os

import fire

from luna.pathology.cli.run_tissue_detection import cli


def test_otsu(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--tile-size",
            str(256),
            "--output-urlpath",
            str(tmp_path),
            "--filter-query",
            "otsu_score > 0.5",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123-filtered.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")


def test_stain(tmp_path, dask_client):
    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--output-urlpath",
            str(tmp_path),
            "--requested-magnification",
            str(5),
            "--tile-size",
            str(256),
            "--filter-query",
            "stain0_score > 0.05",
        ],
    )

    assert os.path.exists(f"{tmp_path}/123-filtered.tiles.parquet")
    assert os.path.exists(f"{tmp_path}/metadata.yml")
