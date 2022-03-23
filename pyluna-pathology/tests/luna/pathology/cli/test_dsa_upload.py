from click.testing import CliRunner
from girder_client import GirderClient

from luna.pathology.cli.dsa_upload import cli


def test_upload(monkeypatch):
    def mock_get(*args, **kwargs):

        if args[1] == "/system/check":
            return {}
        if args[1] == "/item?text=123":
            return [
                {
                    "annotation": {"name": "123.svs"},
                    "_id": "uuid",
                    "_modelType": "annotation",
                }
            ]

    def mock_listCollection(*args, **kwargs):
        return [{"name": "test_collection", "_id": "uuid", "_modelType": "collection"}]

    def mock_listResource(*args, **kwargs):
        return [
            {
                "annotation": {"name": "123.svs"},
                "_id": "uuid",
                "_modelType": "annotation",
            }
        ]

    def mock_put(*args, **kwargs):

        if args == "/annotation?itemId=None":
            return {}
        pass

    def mock_auth(*args, **kwargs):

        if args[1] == "myuser" and args[2] == "mypassword":
            return 0  # success
        else:
            return 1  # Access Error

    monkeypatch.setattr(GirderClient, "get", mock_get)
    monkeypatch.setattr(GirderClient, "authenticate", mock_auth)
    monkeypatch.setattr(GirderClient, "put", mock_put)
    monkeypatch.setattr(GirderClient, "listResource", mock_listResource)
    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "http://localhost:8080/",
            "--annotation_filepath",
            "pyluna-pathology/tests/luna/pathology/cli/testouts/"
            + "Tile-Based_Pixel_Classifier_Inference_123.json",
            "--image_filename",
            "123.svs",
            "--collection_name",
            "test_collection",
            "--username",
            "user",
            "--password",
            "pw",
        ],
    )

    assert result.exit_code == 0
