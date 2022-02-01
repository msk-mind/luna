import os
from click.testing import CliRunner

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