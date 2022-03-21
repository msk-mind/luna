import pytest
from girder_client import GirderClient
from luna.pathology.dsa.dsa_api_handler import (
    get_collection_uuid,
    get_collection_uuid,
    get_annotation_uuid,
    get_collection_metadata,
    get_slide_annotation,
    get_item_uuid,
)


def test_get_collection_uuid(monkeypatch):
    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    gc = GirderClient(apiUrl="http://localhost/api/v1")
    collection_uid = get_collection_uuid(gc, "test-collection")

    assert collection_uid == "c-0001"


def test_get_collection_uuid_missing():

    gc = GirderClient(apiUrl="http://localhost/api/v1")

    with pytest.raises(RuntimeError):
        get_collection_uuid(gc, "test-collection")


def test_get_annotation_uuid(monkeypatch):
    def mock_get(*args, **kwargs):
        if args[1] == "annotation?itemId=123":
            return [
                {
                    "_id": "321",
                    "itemId": "123",
                    "_modelType": "annotation",
                    "annotation": {"name": "regional"},
                }
            ]

        pass

    monkeypatch.setattr(GirderClient, "get", mock_get)

    gc = GirderClient(apiUrl="http://localhost/api/v1")
    annotation_uid = get_annotation_uuid(gc, "123", "regional")

    assert annotation_uid == "321"


def test_get_item_uuid(monkeypatch):
    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    def mock_get(*args, **kwargs):
        if args[1] == "/item?text=123":
            return [
                {
                    "name": "123.svs",
                    "_id": "321",
                    "baseParentId": "c-0001",
                    "_modelType": "annotation",
                }
            ]

        pass

    monkeypatch.setattr(GirderClient, "get", mock_get)

    gc = GirderClient(apiUrl="http://localhost/api/v1")
    item_uid = get_item_uuid(gc, "123.svs", "test-collection")

    assert item_uid == "321"


def test_get_collection_metadata(monkeypatch):
    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    def mock_get(*args, **kwargs):
        if args[1] == "/collection/c-0001":
            return {"meta": {"stylesheet": "stylesheet"}, "_id": "321"}
        pass

    monkeypatch.setattr(GirderClient, "get", mock_get)

    gc = GirderClient(apiUrl="http://localhost/api/v1")

    uid, metadata = get_collection_metadata("test-collection", gc)

    assert uid == "c-0001"
    assert metadata == "stylesheet"


def test_get_slide_annotation(monkeypatch):
    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    def mock_get(*args, **kwargs):
        if args[1] == "/item?text=123":
            return [
                {
                    "name": "123.svs",
                    "_id": "321",
                    "baseParentId": "c-0001",
                    "_modelType": "image",
                }
            ]

        if args[1] == "/annotation?itemId=321":
            return [
                {
                    "name": "123.svs",
                    "_id": "a-321",
                    "baseParentId": "c-0001",
                    "_modelType": "annotation",
                    "annotation": {"name": "regional"},
                }
            ]

        if args[1] == "/annotation/a-321":
            return {
                "name": "123.svs",
                "_id": "a-321",
                "created": "2022/03/07",
                "updated": "2022/03/07",
                "creatorId": "u-123",
                "updatedId": "u-123",
                "annotation": {"name": "regional"},
            }
        if args[1] == "/user/u-123":
            return {
                "login": "user",
                "_id": "a-321",
                "created": "2022/03/07",
                "updated": "2022/03/07",
                "creatorId": "user",
                "updatedId": "user",
                "annotation": {"name": "regional"},
            }
        pass

    monkeypatch.setattr(GirderClient, "get", mock_get)
    gc = GirderClient(apiUrl="http://localhost/api/v1")

    slide_id, metadata, json = get_slide_annotation(
        "123.svs", "regional", "test-collection", gc
    )

    assert slide_id == "123.svs"
    assert metadata["annotation_name"] == "regional"
    assert metadata["user"] == "user"


def test_get_slide_no_annotation(monkeypatch):
    def mock_listCollection(*args, **kwargs):
        return [
            {"name": "test-collection", "_id": "c-0001", "_modelType": "collection"}
        ]

    monkeypatch.setattr(GirderClient, "listCollection", mock_listCollection)

    def mock_get(*args, **kwargs):
        if args[1] == "/item?text=123":
            return [
                {
                    "name": "123.svs",
                    "_id": "321",
                    "baseParentId": "c-0001",
                    "_modelType": "image",
                }
            ]
        pass

    monkeypatch.setattr(GirderClient, "get", mock_get)
    gc = GirderClient(apiUrl="http://localhost/api/v1")

    res = get_slide_annotation("123.svs", "regional", "test-collection", gc)

    assert res is None
