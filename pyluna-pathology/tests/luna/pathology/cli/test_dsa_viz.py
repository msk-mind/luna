from click.testing import CliRunner
import os
from pathlib import Path

from luna.pathology.cli.dsa_viz import cli


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
            "pyluna-pathology/tests/luna/pathology/cli/testdata/test_object_classification.geojson",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata"
            "/stardist_polygon.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts"
        "/StarDist_Segmentations_with_Lymphocyte_Classifications_123.json"
    )
    verify_cleanup(output_file)


def test_stardist_cell():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "stardist-cell",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/test_object_detection.tsv",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata" "/stardist_cell.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts"
        "/Points_of_Classsified_StarDist_Cells_123.json"
    )
    verify_cleanup(output_file)


def test_regional_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "regional-polygon",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/regional_annotation.json",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata"
            "/regional_polygon.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts"
        "/Slideviewer_Regional_Annotations_123.json"
    )
    verify_cleanup(output_file)


def test_bitmask_polygon_invalid():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "bitmask-polygon",
            '{"Tumor": "non/existing/path/to/png.png"}',
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata" "/bitmask_polygon.yml",
        ],
    )
    assert isinstance(result.exception, ValueError)


""" works locally. but times out on circleci
def test_bitmask_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "bitmask-polygon",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata"
            "/bitmask_polygon.yml",
        ],
    )

    assert result.exit_code == 0

    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts"
        "/Full_Res_Tile-Based_Pixel_Classifier_Inference_123.json"
    )
    verify_cleanup(output_file)
"""


def test_heatmap():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "heatmap",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/tile_scores.csv",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata" "/heatmap_config.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts" "/otsu_score_test_123.json"
    )
    verify_cleanup(output_file)


def test_qupath_polygon():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "qupath-polygon",
            "pyluna-pathology/tests/luna/pathology/cli/testdata/region_annotation_results.geojson",
            "-m",
            "pyluna-pathology/tests/luna/pathology/cli/testdata" "/qupath_polygon.yml",
        ],
    )

    assert result.exit_code == 0
    output_file = (
        "pyluna-pathology/tests/luna/pathology/cli/testouts"
        "/Qupath_Pixel_Classifier_Polygons_123.json"
    )
    verify_cleanup(output_file)
