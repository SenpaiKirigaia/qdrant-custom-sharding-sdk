import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

import qdrant_custom_sharding as sdk
from qdrant_custom_sharding.async_sharded_client import AsyncShardingAdmin
from qdrant_custom_sharding.errors import ShardingError
from qdrant_custom_sharding.map_cache import MapCache
from qdrant_custom_sharding.map_stores import (
    FileMapFolder,
    S3MapFolder,
    list_map_collections,
    open_map_folder,
    register_map_folder_scheme,
    unregister_map_folder_scheme,
)
from qdrant_custom_sharding.models import FeatureInfo, MapDoc
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _doc(collection: str) -> MapDoc:
    return MapDoc(
        collection=collection,
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
    )


def test_file_map_folder_lists_only_json_maps(tmp_path: Path):
    missing_dir = tmp_path / "absent"
    assert FileMapFolder(missing_dir).list_collections() == []

    folder = FileMapFolder(tmp_path / "maps")
    folder.store_for("zeta").save(_doc("zeta"))
    folder.store_for("alpha").save(_doc("alpha"))
    (tmp_path / "maps" / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "maps" / "alpha.json.lock").write_text("", encoding="utf-8")
    (tmp_path / "maps" / "nested.json").mkdir()
    assert folder.list_collections() == ["alpha", "zeta"]


def test_s3_map_folder_lists_paginated_collection_maps():
    class _FakeS3:
        def __init__(self) -> None:
            self.calls = []

        def list_objects_v2(self, **kwargs):
            self.calls.append(kwargs)
            if "ContinuationToken" not in kwargs:
                return {
                    "Contents": [
                        {"Key": "maps/zeta.json"},
                        {"Key": "maps/nested/ignored.json"},
                        {"Key": "maps/readme.txt"},
                    ],
                    "IsTruncated": True,
                    "NextContinuationToken": "page-2",
                }
            return {
                "Contents": [{"Key": "maps/alpha.json"}],
                "IsTruncated": False,
            }

    s3 = _FakeS3()
    folder = S3MapFolder("bucket", "maps/", client=s3)
    assert folder.list_collections() == ["alpha", "zeta"]
    assert len(s3.calls) == 2


def test_custom_uri_scheme_registration_and_cleanup():
    created: dict[str, FakeMapFolder] = {}

    def open_memory(uri: str) -> FakeMapFolder:
        folder = FakeMapFolder()
        created[uri] = folder
        return folder

    register_map_folder_scheme("mem", open_memory)
    try:
        opened = open_map_folder("mem://cluster-a/maps")
        assert opened is created["mem://cluster-a/maps"]
    finally:
        unregister_map_folder_scheme("mem")

    # After unregistering, falls back to local path handling.
    assert isinstance(open_map_folder("mem://cluster-a/maps"), FileMapFolder)


def test_register_scheme_rejects_invalid_names():
    with pytest.raises(ValueError, match="scheme"):
        register_map_folder_scheme("", lambda _uri: FakeMapFolder())
    with pytest.raises(ValueError, match="scheme"):
        register_map_folder_scheme("bad scheme!", lambda _uri: FakeMapFolder())


def test_list_map_collections_reports_unsupported_custom_folder():
    class _MinimalFolder:
        def store_for(self, collection: str):
            raise NotImplementedError

    with pytest.raises(ShardingError, match="listing"):
        list_map_collections(_MinimalFolder())


def test_admin_and_async_admin_list_collections():
    folder = FakeMapFolder()
    folder.store_for("users").save(_doc("users"))
    folder.store_for("catalog").save(_doc("catalog"))
    # Accessed but never saved should not show up.
    folder.store_for("empty")
    vectors = FakeVectorStore()

    admin = ShardingAdmin(
        folder,
        lambda name: MapCache(folder.store_for(name), 30, lambda: NOW),
        lambda: NOW,
        lambda _s: None,
        vectors,
    )
    assert admin.list_collections() == ["catalog", "users"]

    async def _run():
        async def _asleep(_s: int):
            return None

        async_admin = AsyncShardingAdmin(
            folder,
            lambda name: MapCache(folder.store_for(name), 30, lambda: NOW),
            lambda: NOW,
            _asleep,
            vectors,
        )
        return await async_admin.list_collections()

    assert asyncio.run(_run()) == ["catalog", "users"]


def test_extension_points_are_exported_on_top_level_package():
    expected = {
        "CutoverResult",
        "FeatureInfo",
        "FileMapFolder",
        "FileMapStore",
        "ListableMapFolder",
        "MapDoc",
        "MapFolder",
        "MapStore",
        "Point",
        "S3MapFolder",
        "S3MapStore",
        "VectorStore",
        "map_from_json",
        "map_to_json",
        "open_map_folder",
        "register_map_folder_scheme",
        "unregister_map_folder_scheme",
    }
    assert expected <= set(sdk.__all__)
    for symbol in expected:
        assert hasattr(sdk, symbol)
