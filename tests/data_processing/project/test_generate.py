import os, shutil, sys
from click.testing import CliRunner

from data_processing.project.generate import cli

project_path = 'tests/data_processing/project/testdata/PRO_12-123'
manifest_path = 'tests/data_processing/project/testdata/PRO_12-123/manifest.yaml'



def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, [
        '-m', 'tests/data_processing/project/testdata/project_manifest.yaml'])
    assert result.exit_code == 0

    assert(project_path)
    assert os.path.exists(manifest_path)

    # clean up
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
