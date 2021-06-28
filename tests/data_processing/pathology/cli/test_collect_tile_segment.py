from click.testing import CliRunner

from data_processing.common.Neo4jConnection import Neo4jConnection
from data_processing.pathology.cli.collect_tile_segment import cli
import shutil

def test_cli(mocker):

    mocker.patch.object(Neo4jConnection, 'query')
    prop_null = [{'mock':'response'}] # Used when only a response is needed, but data doesn't matter
    props1 = [{"id(container)":"id",
                "container.name":"store_123",
                "container.qualified_address":"test::store_123",
                "container.type":"slide",
                "labels(container)":"slide"}]
    props2 = [{"id(container)":"id",
                "container.name":"parquet_store_123",
                "container.qualified_address":"test::parquet_store_123",
                "container.type":"parquet",
                "labels(container)":"slide"}]
    props3 = [{"id(container)":"id",
            "container.name":"store_123",
            "container.qualified_address":"test::store_123",
            "container.type":"slide",
            "labels(container)":"slide",
            "data":{
                "type": "TileScores", "name":"123", 
                "tile_size": 128,
                "magnification": 20,
                "object_bucket": 'bucket-test',
                "object_folder": '123',
                "data": "tests/data_processing/pathology/cli/testdata/data/test/store_123/test_generate_tile_ov_labels/tiles.slice.pil",
                "aux":  "tests/data_processing/pathology/cli/testdata/data/test/store_123/test_generate_tile_ov_labels/address.slice.csv"
            }}]

    props4 = [{"id(container)":"id",
            "container.name":"store_123",
            "container.qualified_address":"test::store_123",
            "container.type":"slide",
            "labels(container)":"slide",
            "data":{
                "type": "WholeSlideImage", "name":"123", 
            }}]


    # Call order goes [ namespace check, data_store set,  namespace check, data_store create,  data_store set,  get TileScores, get WholeSlideImage, put ]
    Neo4jConnection.query.side_effect  = [prop_null, props1, prop_null, prop_null, props2, props3, props4, prop_null]
    Neo4jConnection.query.return_value = Neo4jConnection.query.side_effect

    mocker.patch.object(Neo4jConnection, 'test_connection')

    runner = CliRunner()
    result = runner.invoke(cli, [
        '-a', 'tests/data_processing/pathology/cli/testdata/test_config.yaml',
        '-c', 'test',
        '-s', '123',
        '-m', 'tests/data_processing/pathology/cli/testdata/collect_tile_results.json'])

    assert result.exit_code == 0
    # cleanup
    shutil.rmtree('tests/data_processing/pathology/cli/testdata/data/test/test')

