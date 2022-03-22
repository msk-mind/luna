from click.testing import CliRunner

from girder_client import GirderClient


from luna.pathology.cli.dsa_annotation_etl import cli
import numpy as np


def test_cli(tmp_path, monkeypatch):
    def mock_get(*args, **kwargs):

        if args[1] == "/system/check":
            return {}

        if args[1] == "/collection?text=test_collection":
            return [
                {"name": "test-collection", "_id": "uuid", "_modelType": "collection"}
            ]

        if args[1] == "/item?text=123":
            return [
                {
                    "annotation": {"name": "123.svs"},
                    "_id": "uuid",
                    "_modelType": "annotation",
                }
            ]

        pass

    def mock_put(*args, **kwargs):

        if args == "/annotation?itemId=None":
            return {}

        pass

    def mock_auth(*args, **kwargs):

        if args[1] == "myuser" and args[2] == "mypassword":
            return 0  # success
        else:
            return 1  # Access Error

    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    def mock_listResource(*args, **kwargs):
        return [
            {
                "name": "test-resource",
                "_id": "r-0001",
                "largeImage": np.nan,
                "_modelType": "resource",
            }
        ]  # A way to mock 0 slides

    monkeypatch.setattr(GirderClient, "get", mock_get)
    monkeypatch.setattr(GirderClient, "authenticate", mock_auth)
    monkeypatch.setattr(GirderClient, "put", mock_put)
    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)
    monkeypatch.setattr(GirderClient, "listResource", mock_listResource)

    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "http://localhost:8080/api/v1",
            "-u",
            "username",
            "-p",
            "password",
            "-c",
            "test-collection",
            "-a",
            "test-annotation",
            "-o",
            tmp_path,
        ],
    )

    assert result.exit_code == 0
