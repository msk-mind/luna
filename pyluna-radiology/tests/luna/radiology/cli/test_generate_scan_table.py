import os
import yaml
import pandas as pd
from click.testing import CliRunner
from luna.radiology.cli.generate_scan_table import *


def test_cli(tmpdir):

    scan_table_path = str(tmpdir) + "/scan_table"
    runner = CliRunner()
    result = runner.invoke(cli,
        ['--raw_data_path', 'pyluna-radiology/tests/luna/radiology/cli/testdata/',
         '--mapping_csv_path', 'pyluna-radiology/tests/luna/radiology/cli/testdata/series_mapping.csv',
         '--output_dir', tmpdir,
         '--scan_table_path', scan_table_path,
         '--npartitions', 1,
         '--subset'])
    print(result.exit_code)
    print(result.exc_info)

    assert result.exit_code == 0
    assert os.path.exists(str(tmpdir)+"/123")
    with open (str(tmpdir) + '/metadata.yml') as fp:
        metadata = yaml.safe_load(fp)
        assert metadata['table_path']== scan_table_path

    df = pd.read_parquet(scan_table_path)
    print(df)
    assert len(df) == 1
    assert set(["record_uuid", "accession_number", "series_number", "dim", "path", "subset_path", "create_ts"])\
           == set(df.columns)
