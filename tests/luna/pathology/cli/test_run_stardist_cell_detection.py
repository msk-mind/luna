import io
import os

import fire

import docker
from luna.pathology.cli.run_stardist_cell_detection import stardist_simple

tmppath = "tests/testdata/pathology/stardist_cell_detection"


class MockContainer(object):
    def logs(*args, **kwargs):
        return io.StringIO(None)


class MockContainerCollection(object):
    def run(*args, **kwargs):
        return MockContainer()


class MockClient(object):
    @property
    def containers(*args, **kwargs):
        return MockContainerCollection()


def test_cli(monkeypatch):
    def mock_container_collection(*args, **kwargs):
        return MockContainerCollection()

    def mock_container(*args, **kwargs):
        return MockContainer()

    def mock_client(*args, **kwargs):
        return MockClient()

    monkeypatch.setattr(docker, "from_env", mock_client)
    monkeypatch.setattr(
        docker.client.DockerClient, "containers", mock_container_collection
    )
    monkeypatch.setattr(
        docker.models.containers.ContainerCollection, "run", mock_container
    )
    monkeypatch.setattr(docker.models.containers.Container, "logs", mock_container)

    fire.Fire(
        stardist_simple,
        [
            "--slide-urlpath",
            "tests/testdata/pathology/123.svs",
            "--output-urlpath",
            str(tmppath),
            "--cell-expansion-size",
            "8",
            "--image-type",
            "BRIGHTFIELD_H_DAB",
        ],
    )

    assert os.path.exists(f"{tmppath}/cell_detections.tsv")
