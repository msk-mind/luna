import os
from click.testing import CliRunner
import pytest 

from luna.pathology.cli.dsa_annotation_etl import cli

def test_cli(tmp_path):

    runner = CliRunner()

    result = runner.invoke(cli, [
        'http://localhost:8080/api/v1',
        '-u', 'username',
        '-p', 'password',
        '-c', 'some-collection',
        '-a', 'some-annotation',
        '-o', tmp_path,
    ])    

    assert f"{result.exception}" == "Connection to DSA endpoint failed."

