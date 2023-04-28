import os

import fire
import pandas as pd

from luna.radiology.cli.generate_scan_table import generate_scan_table


def test_cli(tmpdir):
    scan_table_path = str(tmpdir) + "/scan_table"
    fire.Fire(
        generate_scan_table,
        [
            "--raw_data_path",
            "tests/testdata/radiology/",
            "--mapping_csv_path",
            "tests/testdata/radiology/series_mapping.csv",
            "--output_dir",
            str(tmpdir),
            "--scan_table_path",
            str(scan_table_path),
            "--npartitions",
            "1",
            "--subset",
        ],
    )

    assert os.path.exists(str(tmpdir) + "/123")
    # with open(str(tmpdir) + "/metadata.yml") as fp:
    # metadata = yaml.safe_load(fp)
    # assert metadata["table_path"] == scan_table_path

    df = pd.read_parquet(scan_table_path)
    print(df)
    assert len(df) == 1
    assert set(
        [
            "record_uuid",
            "accession_number",
            "series_number",
            "dim",
            "path",
            "subset_path",
            "create_ts",
        ]
    ) == set(df.columns)
