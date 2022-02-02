import os
from click.testing import CliRunner

import pandas as pd
from luna.pathology.cli.generate_tiles import cli


def test_cli(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, [
        'pyluna-pathology/tests/luna/pathology/cli/testdata/data/123.svs',
        '-o', tmp_path,
        '-rts', 256,
        '-rmg', 5,
        '-nc', 1,
        '-bx', 1])

    assert result.exit_code == 0
    assert os.path.exists(f"{tmp_path}/123.tiles.csv")
    assert os.path.exists(f"{tmp_path}/123.tiles.h5")

    df = pd.read_csv(f"{tmp_path}/123.tiles.csv")

    REQ_COLUMNS = set(['address', 'tile_size', 'tile_units', 'x_coord', 'y_coord'])

    assert set(df.columns).intersection(REQ_COLUMNS) == REQ_COLUMNS
