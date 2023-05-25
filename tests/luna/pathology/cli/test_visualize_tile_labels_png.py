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
