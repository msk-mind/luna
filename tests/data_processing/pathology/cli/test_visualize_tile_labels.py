import subprocess
import shutil
from click.testing import CliRunner

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.pathology.cli.visualize_tile_labels import cli


def test_cli(mocker):

    mocker.patch.object(Neo4jConnection, 'query')
    prop_null = [{'mock':'response'}]
    props1 = [{"id(container)":"id",
             "container.name":"store_123",
             "container.qualified_address":"test::store_123",
             "container.type":"slide",
             "labels(container)":"slide"}]
    props2 = [{"id(container)":"id",
             "container.name":"store_123",
             "container.qualified_address":"test::store_123",
             "container.type":"slide",
             "dzlabels(container)":"slide",
             "data":{
                 "type":"WholeSlideImage", "name":"123", 
                 "data":"tests/data_processing//testdata/data/test-project/wsi/123.svs"
            }}]
    props3 = [{"id(container)":"id",
            "container.name":"store_123",
            "container.qualified_address":"test::store_123",
            "container.type":"slide",
            "labels(container)":"slide",
            "data":{
                "type": "TileScores", "name":"123", 
                "tile_size": 128,
                "magnification": 20,
                "scale_factor": 8,
                "data": "tests/data_processing/pathology/cli/testdata/data/test/store_123/test_generate_tile_ov_labels/address.slice.csv"
            }}]
    # Call order goes [ namespace check, data_store check, get WholeSlideImage, get TileScores, put ]
    Neo4jConnection.query.side_effect  = [[prop_null], props1, props2, props3, [prop_null]]
    Neo4jConnection.query.return_value = Neo4jConnection.query.side_effect

    mocker.patch.object(Neo4jConnection, 'test_connection')
    mocker.patch.object(subprocess, "run", return_value='tests/data_processing/pathology/cli/dsa/testouts/Tile-Based_Pixel_Classifier_Inference_123.json')

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-c', 'test',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/visualize_tile_labels.json'])

    assert result.exit_code == 0
    shutil.rmtree('tests/data_processing/pathology/cli/testdata/data_staging')
