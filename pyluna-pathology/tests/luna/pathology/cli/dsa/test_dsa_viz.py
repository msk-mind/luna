from click.testing import CliRunner
import os
from pathlib import Path

from girder_client import GirderClient

from luna.pathology.cli.dsa.dsa_viz import cli
from luna.pathology.cli.dsa.dsa_upload import cli as upload


def verify_cleanup(output_file):
    assert os.path.exists(output_file)
    # cleanup
    os.remove(output_file)
    os.remove(str(Path(output_file).parent) + "/metadata.yml")


def test_stardist_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stardist-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/stardist_polygon.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/StarDist_Segmentations_with_Lymphocyte_Classifications_123.json"
    )
    verify_cleanup(output_file)


def test_stardist_cell():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stardist-cell",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/stardist_cell.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/Points_of_Classsified_StarDist_Cells_123.json"
    )
    verify_cleanup(output_file)


def test_regional_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "regional-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/regional_polygon.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/Slideviewer_Regional_Annotations_123.json"
    )
    verify_cleanup(output_file)


def test_bitmask_polygon_invalid():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "bitmask-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/bitmask_polygon.yml",
            "-i",
            {"Tumor": "non/existing/path/to/png.png"},
        ],
    )
    assert isinstance(result.exception, ValueError)


def test_bitmask_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "bitmask-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/bitmask_polygon.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/Full_Res_Tile-Based_Pixel_Classifier_Inference_123.json"
    )
    verify_cleanup(output_file)


def test_heatmap():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "heatmap",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/heatmap_config.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/otsu_score_test_123.json"
    )
    verify_cleanup(output_file)


def test_qupath_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "qupath-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/qupath_polygon.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts"
        "/Qupath_Pixel_Classifier_Polygons_123.json"
    )
    verify_cleanup(output_file)


def test_upload(monkeypatch):
    def mock_get(*args, **kwargs):

        if args[1] == "/system/check":
            return {}

        if args[1] == "/collection?text=test_collection":
            return [
                {"name": "test_collection", "_id": "uuid", "_modelType": "collection"}
            ]

        if args[1] == "/item?text=123":
            return [
                {
                    "annotation": {"name": "123.svs"},
                    "_id": "uuid",
                    "_modelType": "annotation",
                }
            ]

        pass

    def mock_put(*args, **kwargs):

        if args == "/annotation?itemId=None":
            return {}

        pass

    def mock_auth(*args, **kwargs):

        if args[1] == "myuser" and args[2] == "mypassword":
            return 0  # success
        else:
            return 1  # Access Error

    monkeypatch.setattr(GirderClient, "get", mock_get)
    monkeypatch.setattr(GirderClient, "authenticate", mock_auth)
    monkeypatch.setattr(GirderClient, "put", mock_put)

    runner = CliRunner()
    result = runner.invoke(
        upload,
        [
            "-c",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata/"
            + "dsa_config.yml",
            "-d",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testdata"
            "/bitmask_polygon_upload.yml",
        ],
    )

    assert result.exit_code == 0
