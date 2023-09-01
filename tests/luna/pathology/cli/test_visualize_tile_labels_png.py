import os

import fire
import numpy as np
import pandas as pd

from luna.pathology.cli.visualize_tile_labels_png import cli


def test_viz(tmp_path):
    df = pd.read_csv("tests/testdata/pathology/generate_tiles/123/123.tiles.csv")
    df["random"] = np.random.rand(len(df))
    df.to_parquet(f"{tmp_path}/input_tiles.parquet")

    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--tiles-urlpath",
            f"{tmp_path}/input_tiles.parquet",
            "--output-urlpath",
            str(tmp_path),
            "--plot_labels",
            "random",
            "--requested-magnification",
            "5",
        ],
    )

    assert os.path.exists(f"{tmp_path}/tile_scores_and_labels_visualization_random.png")


def test_viz_s3(s3fs_client):
    s3fs_client.mkdirs("viz", exist_ok=True)
    s3fs_client.put("tests/testdata/pathology/123.svs", "viz/test/")

    df = pd.read_csv("tests/testdata/pathology/generate_tiles/123/123.tiles.csv")
    df["random"] = np.random.rand(len(df))
    df.to_parquet(
        "s3://viz/test/input_tiles.parquet",
        storage_options={
            "key": "",
            "secret": "",
            "endpoint_url": s3fs_client.client_kwargs["endpoint_url"],
        },
    )

    fire.Fire(
        cli,
        [
            "--slide-urlpath",
            "s3://viz/test/123.svs",
            "--tiles-urlpath",
            "s3://viz/test/input_tiles.parquet",
            "--output-urlpath",
            "s3://viz/out/",
            "--plot_labels",
            "random",
            "--requested-magnification",
            "5",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists(
        "s3://viz/out/tile_scores_and_labels_visualization_random.png"
    )
