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
