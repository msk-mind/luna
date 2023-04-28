import os

import fire
import pandas as pd
from dask.distributed import Client

from luna.pathology.cli.extract_kfunction_statistics import cli


def test_cli(tmp_path):
    Client()
    fire.Fire(
        cli,
        [
            "--input_cell_objects_url",
            "tests/testdata/pathology/test_tile_stats.parquet",
            "--output_url",
            str(tmp_path),
            "--intensity_label",
            "Centroid X µm",
            "--radius",
            str(160.0),
            "--tile_stride",
            str(300),
            "--tile_size",
            str(300),
        ],
    )

    assert os.path.exists(f"{tmp_path}/test_tile_stats_kfunction_supertiles.parquet")

    df = pd.read_parquet(f"{tmp_path}/test_tile_stats_kfunction_supertiles.parquet")
    assert "ikfunction_r160.0_stainCentroid_X_µm" in df.columns
    assert df["ikfunction_r160.0_stainCentroid_X_µm_norm"].values[0] == 1.0
