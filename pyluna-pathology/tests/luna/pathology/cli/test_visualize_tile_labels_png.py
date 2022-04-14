import os
from click.testing import CliRunner

from luna.pathology.cli.visualize_tile_labels_png import cli

import pandas as pd
import numpy as np


def test_viz(tmp_path):
    df = pd.read_csv("pyluna-pathology/tests/luna/pathology/cli/testdata/data/generate_tiles/123/123.tiles.csv")
    df['random'] = np.random.rand(len(df))
    df.to_parquet(f"{tmp_path}/input_tiles.parquet")

    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs',
        f"{tmp_path}/input_tiles.parquet",
        '-o', tmp_path,
        '-pl', 'random',
        '-rmg', 5])

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/tile_scores_and_labels_visualization_random.png")

