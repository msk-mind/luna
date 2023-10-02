import os
from pathlib import Path

import fire
import pytest

from luna.pathology.cli.dsa_viz import (  # bmp_polygon,
    bitmask_polygon,
    heatmap,
    qupath_polygon,
    regional_polygon,
    stardist_cell,
    stardist_polygon,
)



def test_stardist_polygon_s3(s3fs_client):
    s3fs_client.mkdirs("dsatest", exist_ok=True)
    s3fs_client.put(
        "tests/testdata/pathology/test_object_classification.geojson",
        "s3://dsatest/test/",
    )
    fire.Fire(
        stardist_polygon,
        [
            "--local_config",
            "tests/testdata/pathology/stardist_polygon_s3.yml",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists(
        "s3://dsatest/out/StarDist_Segmentations_with_Lymphocyte_Classifications_123.json"
    )


def test_stardist_polygon(tmp_path):
    fire.Fire(
        stardist_polygon,
        [
            "--local_config",
            "tests/testdata/pathology/stardist_polygon.yml",
            "--output_urlpath", str(tmp_path)
        ],
    )

    output_file = tmp_path / "StarDist_Segmentations_with_Lymphocyte_Classifications_123.json"
    assert os.path.exists(output_file)


def test_stardist_cell(tmp_path):
    fire.Fire(
        stardist_cell,
        [
            "--local_config",
            "tests/testdata/pathology/stardist_cell.yml",
            "--output_urlpath", str(tmp_path)
        ],
    )

    output_file = tmp_path / "Points_of_Classsified_StarDist_Cells_123.json"
    assert os.path.exists(output_file)


def test_regional_polygon(tmp_path):
    fire.Fire(
        regional_polygon,
        [
            "--local_config",
            "tests/testdata/pathology" "/regional_polygon.yml",
            "--output_urlpath", str(tmp_path)
        ],
    )

    output_file = tmp_path / "Slideviewer_Regional_Annotations_123.json"
    assert os.path.exists(output_file)


def test_bitmask_polygon_invalid():
    with pytest.raises(Exception):
        fire.Fire(
            bitmask_polygon,
            [
                "--input_map",
                '{"Tumor": "non/existing/path/to/png.png"}',
                "--local_config",
                "tests/testdata/pathology" "/bitmask_polygon.yml",
            ],
        )


""" doesn't work on github actions
def test_bitmask_polygon():
    fire.Fire(
        bitmask_polygon,
        [
            "--local_config",
            "tests/testdata/pathology" "/bitmask_polygon.yml",
        ],
    )

    output_file = (
        "tests/luna/pathology/cli/testouts"
        "/Full_Res_Tile-Based_Pixel_Classifier_Inference_123.json"
    )
    verify_cleanup(output_file)
"""


def test_heatmap(tmp_path):
    fire.Fire(
        heatmap,
        [
            "tests/testdata/pathology/tile_scores.parquet",
            "--column",
            "otsu_score",
            "--local_config",
            "tests/testdata/pathology" "/heatmap_config.yml",
            "--output_urlpath", str(tmp_path)
        ],
    )

    output_file = tmp_path / "otsu_score_test_123.json"
    assert os.path.exists(output_file)


def test_qupath_polygon(tmp_path):
    fire.Fire(
        qupath_polygon,
        [
            "tests/testdata/pathology/region_annotation_results.geojson",
            "--local_config",
            "tests/testdata/pathology" "/qupath_polygon.yml",
            "--output_urlpath", str(tmp_path)
        ],
    )

    output_file = tmp_path / "Qupath_Pixel_Classifier_Polygons_123.json"
    assert os.path.exists(output_file)
