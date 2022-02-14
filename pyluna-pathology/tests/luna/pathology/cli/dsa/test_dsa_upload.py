from click.testing import CliRunner
from girder_client import GirderClient

from luna.pathology.cli.dsa.dsa_upload import cli


def test_upload(monkeypatch):
    def mock_get(*args, **kwargs):

        if args[1] == "/system/check":
            return {}

        if args[1] == "/collection?text=test_collection":
            return [
                {"name": "test_collection", "_id": "uuid", "_modelType": "collection"}
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

    monkeypatch.setenv("DSA_USERNAME", "user")
    monkeypatch.setenv("DSA_PASSWORD", "pw")
    monkeypatch.setattr(GirderClient, "get", mock_get)
    monkeypatch.setattr(GirderClient, "authenticate", mock_auth)
    monkeypatch.setattr(GirderClient, "put", mock_put)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "http://localhost:8080/",
            "--annotation_filepath",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/"
            + "Tile-Based_Pixel_Classifier_Inference_123.json",
            "--image_filename",
            "123.svs",
            "--collection_name",
            "test_collection",
        ],
    )

    assert result.exit_code == 0


def test_dsa_upload_missing_creds():

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "http://localhost:8080/",
            "--annotation_filepath",
            "pyluna-pathology/tests/luna/pathology/cli/dsa/testouts/"
            + "Tile-Based_Pixel_Classifier_Inference_123.json",
            "--image_filename",
            "123.svs",
            "--collection_name",
            "test_collection",
        ],
    )
    # Expect a KeyError when DSA_USERNAME, DSA_PASSWORD is not set
    assert isinstance(result.exception, KeyError)
