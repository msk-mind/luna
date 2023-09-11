import fire
import pandas as pd

from luna.pathology.cli.generate_tile_labels import cli


def test_cli(tmp_path):
    fire.Fire(
        cli,
        [
            "--annotation-urlpath",
            "tests/testdata/pathology/dsa_annots/slide_annotation_dataset_lung-project_Tissue-clf.parquet",
            "--tiles-urlpath",
            "tests/testdata/pathology/save_tiles/123/123.tiles.parquet",
            "--output-urlpath",
            str(tmp_path),
            "--slide-id",
            "123",
        ],
    )

    out_tile = (
        pd.read_parquet(f"{tmp_path}/123.regional_label.tiles.parquet")
        .reset_index()
        .set_index("address")
    )

    assert out_tile.loc["x1_y1_z10.0", "regional_label"] == "Other"
    assert out_tile.loc["x3_y4_z10.0", "regional_label"] == "Tumor"


def test_cli_s3(s3fs_client):
    s3fs_client.mkdirs("tilelabel", exist_ok=True)
    s3fs_client.put(
        "tests/testdata/pathology/dsa_annots/slide_annotation_dataset_lung-project_Tissue-clf.parquet",
        "tilelabel/test/",
    )
    s3fs_client.put(
        "tests/testdata/pathology/save_tiles/123/123.tiles.parquet", "tilelabel/test/"
    )
    fire.Fire(
        cli,
        [
            "--annotation-urlpath",
            "s3://tilelabel/test/slide_annotation_dataset_lung-project_Tissue-clf.parquet",
            "--tiles-urlpath",
            "s3://tilelabel/test/123.tiles.parquet",
            "--output-urlpath",
            "s3://tilelabel/out/",
            "--slide-id",
            "123",
            "--storage_options",
            "{'key': '', 'secret': '', 'client_kwargs': {'endpoint_url': '"
            + s3fs_client.client_kwargs["endpoint_url"]
            + "'}}",
        ],
    )

    assert s3fs_client.exists("tilelabel/out/123.regional_label.tiles.parquet")
