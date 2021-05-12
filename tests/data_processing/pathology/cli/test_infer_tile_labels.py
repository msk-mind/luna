from click.testing import CliRunner

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.pathology.cli.infer_tile_labels import cli


def test_cli(mocker):

    mocker.patch.object(Neo4jConnection, 'query')
    props = {"id(container)":"id",
             "container.name":"name",
             "container.qualified_address":"address",
             "container.type":"slide",
             "labels(container)":"tag",
             "data":{"type":"WholeSlideImage", "name":"123"}}
    Neo4jConnection.query.return_value = [props]

    mocker.patch.object(Neo4jConnection, 'test_connection')

    runner = CliRunner(env={'MIND_GPFS_DIR':'tests/data_processing/pathology/cli/testdata'})
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-c', 'test',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/infer_tile_labels_resnet18.json'])

    # No longer error gracefully -- can update tests with proper data and they'll work
    assert result.exit_code == 1
