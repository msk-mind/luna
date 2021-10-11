import pytest
from minio import Minio
import pyarrow.parquet as pq

from luna.api.radiologyPreprocessingLibrary import app


@pytest.fixture
def client():
    # setup flask api client for testing
    app.app.config["OBJECT_URI"] = "mockuri:1000"
    app.app.config["OBJECT_USER"] = "mockuser"
    app.app.config["OBJECT_PASSWORD"] = "mockpassword"

    with app.app.test_client() as client:
        yield client

class GetObjectResponse:

    metadata = {'Accept-Ranges': 'bytes',
                'Content-Security-Policy': 'block-all-mixed-content',
                'Content-Type': 'application/xml'}

def test_app_post(client, monkeypatch):

    def mock_bucket(*args, **kwargs):
        return False

    monkeypatch.setattr(Minio, "bucket_exists", mock_bucket)
    monkeypatch.setattr(Minio, "make_bucket", mock_bucket)
    monkeypatch.setattr(pq, "write_table", mock_bucket)

    data = {"paths": ["pyluna-core/tests/luna/api/radiologyPreprocessingLibrary/testdata/1.dcm",
                      "pyluna-core/tests/luna/api/radiologyPreprocessingLibrary/testdata/2.dcm"],
            "width": 512,
            "height": 512}

    response = client.post('/radiology/images/project_id/scan_id', json=data)

    print(response.json)
    assert 200 == response.status_code
    assert response.json['message'].startswith('Parquet created at ')


def test_app_post_missing_input(client):

    response = client.post('/radiology/images/project_id/scan_id')

    assert 400 == response.status_code
    assert response.json['message'].startswith('Missing ')


def test_app_post_bad_input(client):

    data = {"dicom_paths": ["pyluna-core/tests/luna/api/radiologyPreprocessingLibrary/testdata/1.dcm",
                      "pyluna-core/tests/luna/api/radiologyPreprocessingLibrary/testdata/2.dcm"],
            "width": 512}

    response = client.post('/radiology/images/project_id/scan_id', json=data)

    assert 400 == response.status_code
    assert response.json['message'].startswith('Missing ')


def test_app_get(client, monkeypatch):

    def mock_get(*args, **kwargs):
        return GetObjectResponse()

    monkeypatch.setattr(Minio, "fget_object", mock_get)

    data = {"output_location": "src/api/tests/api/test.parquet"}

    response = client.get('/radiology/images/project_id/scan_id', json=data)

    assert 200 == response.status_code
    assert response.json['message'].startswith('Downloaded object ')


def test_app_get_missing_input(client):

    response = client.get('/radiology/images/project_id/scan_id')

    assert 400 == response.status_code
    assert 'Missing expected params.' == response.json['message']



def test_app_delete(client, monkeypatch):

    def mock_get(*args, **kwargs):
        return GetObjectResponse()

    monkeypatch.setattr(Minio, "remove_object", mock_get)

    response = client.delete('/radiology/images/project_id/scan_id')

    assert 200 == response.status_code
    assert response.json['message'].startswith('Removed object')

