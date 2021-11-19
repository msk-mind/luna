from click.testing import CliRunner
import os

from girder_client import GirderClient

from luna.pathology.cli.dsa.dsa_viz import cli
from luna.pathology.cli.dsa.dsa_upload import cli as upload


def test_stardist_polygon():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "stardist-polygon",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/stardist_polygon.yml"])

    assert result.exit_code == 0
    output_file = "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/StarDist_Segmentations_with_Lymphocyte_Classifications_123.json"
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)

def test_stardist_cell():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "stardist-cell",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/stardist_cell.yml"])

    assert result.exit_code == 0

    output_file = "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/Points_of_Classsified_StarDist_Cells_123.json"
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)


def test_regional_polygon():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "regional-polygon",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/regional_polygon.yml"])

    assert result.exit_code == 0

    output_file = "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/Slideviewer_Regional_Annotations_123.json"
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)


def test_heatmap():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "heatmap",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/heatmap_config.yml"])

    assert result.exit_code == 0
    output_file = "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/otsu_score_test_123.json"
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)

def test_qupath_polygon():

    runner = CliRunner()
    result = runner.invoke(cli,
                           ["-s", "qupath-polygon",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/qupath_polygon.yml"])

    print(result.exc_info)
    assert result.exit_code == 0
    output_file = "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/Qupath_Pixel_Classifier_Polygons_123.json"
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)

def test_upload(monkeypatch):


    def mock_get(*args, **kwargs):
        if args == '/system/check':
            return {}

        if args == '/collection?text=test_collection':
            return [{'name': 'test_collection', '_id': 'uuid', '_modelType':'collection'}]

        if args == '/item?text=123':
            return [{"annotation": {"name": "123.svs"}, "_id": "uuid", "_modelType":"annotation"}]

        pass
    
    def mock_put(*args, **kwargs):

        if args == '/annotation?itemId=None':
            return {}

        pass
    
    def mock_auth(*args, **kwargs):
        if kwargs['username'] is not None and kwargs['password'] is not None:
            return 0 # success
        else:
            return 1 # Access Error 

    monkeypatch.setattr(GirderClient, "get", mock_get)
    monkeypatch.setattr(GirderClient, "authenticate", mock_auth)
    monkeypatch.setattr(GirderClient, "put", mock_put)

    runner = CliRunner()
    result = runner.invoke(upload,
                           ["-c", "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/dsa_config.yml",
                            "-d","pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/bitmask_polygon_upload.yml"])

    assert result.exit_code == 0
