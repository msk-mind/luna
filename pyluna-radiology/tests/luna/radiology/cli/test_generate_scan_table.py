import os
import shutil

import pytest
import pandas as pd
from click.testing import CliRunner
from luna.radiology.cli.generate_scan_table import *

scan_table_path = 'pyluna-radiology/tests/luna/radiology/cli/testdata/scan_table'
method_param_path = scan_table_path + "_params.json"
output_dir = 'pyluna-radiology/tests/luna/radiology/cli/testdata/scans'

def teardown():
    # cleanup
    shutil.rmtree(output_dir)
    os.remove(scan_table_path)
    os.remove(method_param_path)


def test_cli():

    runner = CliRunner()
    result = runner.invoke(cli,
        ['--raw_data_path', 'pyluna-radiology/tests/luna/radiology/cli/testdata/',
         '--mapping_csv_path', 'pyluna-radiology/tests/luna/radiology/cli/testdata/series_mapping.csv',
         '--output_dir', output_dir,
         '--scan_table_path', scan_table_path,
         '--npartitions', 1])
    print(result.exit_code)
    print(result.exc_info)

    assert result.exit_code == 0
    assert os.path.exists(method_param_path)
    assert os.path.exists(os.path.join(output_dir, "123"))

    df = pd.read_parquet(scan_table_path)
    print(df)
    assert len(df) == 1
    assert set(["record_uuid", "accession_number", "series_number", "dim", "path", "subset_path", "create_ts"])\
           == set(df.columns)
