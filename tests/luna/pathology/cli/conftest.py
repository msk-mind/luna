import pytest
from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster()
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()
