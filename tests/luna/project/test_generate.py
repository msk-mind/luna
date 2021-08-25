import os, shutil, sys
from click.testing import CliRunner

from luna.project.generate import cli

project_path = 'tests/luna/project/testdata/PRO_12-123'
manifest_path = 'tests/luna/project/testdata/PRO_12-123/manifest.yaml'



def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli, [
        '-m', 'tests/luna/project/testdata/project_manifest.yaml'])
    assert result.exit_code == 0

    assert(project_path)
    assert os.path.exists(manifest_path)

    # clean up
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
