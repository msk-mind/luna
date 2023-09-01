import os

import pytest
import s3fs
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster()
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope="module")
def s3fs_client():
    localstack_endpoint = os.getenv(
        "LOCALSTACK_ENDPOINT_URL", default="http://localhost:4566"
    )
    return s3fs.core.S3FileSystem(
        key="", secret="", client_kwargs={"endpoint_url": localstack_endpoint}
    )
