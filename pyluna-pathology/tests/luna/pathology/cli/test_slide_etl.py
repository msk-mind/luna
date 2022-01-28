import pytest, os
from click.testing import CliRunner
from luna.pathology.cli.slide_etl import cli
import pandas as pd

def test_slide_etl(tmp_path):

    runner = CliRunner()

    table_location = os.path.join(tmp_path, 'tables')
    store_location = os.path.join(tmp_path, 'data')

    os.makedirs(store_location)

    result = runner.invoke(cli, [
            'pyluna-pathology/tests/luna/pathology/testdata/data/test-project/wsi',
            '-o', table_location,
            '--store_url', f'file://{store_location}',
            '--num_cores', 4,
            '--project_name', 'TEST-00-000',
            '--comment', 'Test ingestion',
            ])

    # No longer error gracefully -- can update tests with proper data and they'll work
    assert result.exit_code ==0

    df = pd.read_parquet(os.path.join(table_location, f"slide_ingest_TEST-00-000.parquet"))

    assert df.loc['123', 'size'] ==1938955
    assert df.loc['123', 'aperio.AppMag'] == 20
    assert df.loc['123', 'data_url'] == "file://" + os.path.join(store_location, 'pathology/TEST-00-000/slides/123.svs')
