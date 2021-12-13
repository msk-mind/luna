from click.testing import CliRunner

from luna.pathology.cli.infer_tile_labels import cli


def test_cli():

    runner = CliRunner()

    result = runner.invoke(cli, [
            '-i', 'pyluna-pathology/tests/luna/pathology/cli/testdata/123/data',
            '-o', './tmp/123',
            '-rn', 'msk-mind/luna-ml',
            '-tn', 'tissue_tile_net_transform',
            '-mn', 'tissue_tile_net_model_5_class',
            '-wt', 'main:tissue_net_2021-01-19_21.05.24-e17.pth',
            ])

    # No longer error gracefully -- can update tests with proper data and they'll work
    assert result.exit_code == 1
