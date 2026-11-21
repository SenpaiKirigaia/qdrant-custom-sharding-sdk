import asyncio
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import grpc
import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from qdrant_custom_sharding import async_services
from qdrant_custom_sharding.async_sharded_client import AsyncShardedQdrantClient
from qdrant_custom_sharding.errors import MapUnavailable, ShardingError
from qdrant_custom_sharding.map_cache import MapCache, Reason
from qdrant_custom_sharding.map_stores import FileMapStore
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc
from qdrant_custom_sharding.router import plan_writes
from qdrant_custom_sharding.services import (
    alias_feature,
    apply_rebalance,
    create_sharded_collection,
    cutover,
    migrate,
    open_dual,
    resume_move,
    split_targets,
    start_move,
)
from qdrant_custom_sharding.sharded_client import ShardedQdrantClient
from qdrant_custom_sharding.sharding_warnings import FanOutWarning, FeatureMovedWarning
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore
from tests.local_qdrant import compatibility_kwargs

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _doc(feature_type: str = "keyword") -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        feature_type=feature_type,
        slots=["s0", "s1"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0, "s1": 0},
    )


class _Record:
    def __init__(self, point_id, payload, vector=None):
        self.id = point_id
        self.payload = payload
        self.vector = [0.1, 0.2] if vector is None else vector


def test_file_map_store_creates_missing_parent_directory(tmp_path: Path):
    nested = tmp_path / "deep" / "maps" / "items.json"
    store = FileMapStore(nested)
    store.save(_doc())
    assert nested.exists()
    assert store.load() == _doc()


def test_file_map_store_corrupt_json_raises_map_unavailable(tmp_path: Path):
    path = tmp_path / "items.json"
    store = FileMapStore(path)
    store.save(_doc())
    cache = MapCache(store, 30, lambda: NOW)
    cached, reason = cache.get()
    assert cached is not None
    assert reason is None

    path.write_text("{broken json", encoding="utf-8")
    with pytest.raises(MapUnavailable):
        store.load()

    cache.expire()
    stale, reason = cache.get()
    assert stale == cached
    assert reason is Reason.stale_map


def test_map_cache_serializes_concurrent_loads():
    store = FakeMapStore()
    store.save(_doc())
    loads = 0
    gate = threading.Event()
    original = store.load

    def slow_load() -> MapDoc:
        nonlocal loads
        loads += 1
        gate.wait(timeout=0.05)
        return original()

    store.load = slow_load
    cache = MapCache(store, 30, lambda: NOW)
    results = []

    def worker() -> None:
        results.append(cache.get())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join()

    assert loads == 1
    assert all(doc is not None and reason is None for doc, reason in results)


@pytest.mark.parametrize("bad_value", [None, True, False, ["books"], {"nested": 1}, 3.14])
def test_plan_writes_rejects_invalid_feature_values(bad_value):
    doc = _doc()
    point = PointStruct(id=1, vector=[0.1], payload={"feature": bad_value})
    with pytest.raises(ValueError, match="feature"):
        plan_writes(doc, [point])


def test_plan_writes_rejects_non_integer_for_integer_feature():
    doc = _doc(feature_type="integer")
    point = PointStruct(id=1, vector=[0.1], payload={"feature": "not-a-number"})
    with pytest.raises(ValueError, match="integer"):
        plan_writes(doc, [point])


def test_boundary_validations_reject_non_positive_slots_and_negative_grace():
    store = FakeMapStore()
    vectors = FakeVectorStore()
    with pytest.raises(ValueError, match="n_slots"):
        create_sharded_collection(
            store, vectors, "items", "feature", "keyword", n_slots=0
        )

    doc = _doc()
    with pytest.raises(ValueError, match="negative"):
        doc.least_loaded(-1, set())

    with pytest.raises(ValueError, match="partitions"):
        split_targets(doc, "books", 0)
    with pytest.raises(ValueError, match="partitions"):
        split_targets(doc, "books", -2)

    store.save(doc)
    with pytest.raises(ValueError, match="grace_seconds"):
        start_move(store, "books", ["s1"], NOW, -1)
    with pytest.raises(ValueError, match="grace_seconds"):
        open_dual(store, "books", NOW, -1)
    with pytest.raises(ValueError, match="grace_seconds"):
        cutover(store, vectors, "books", NOW, -1)
    with pytest.raises(ValueError, match="grace_seconds"):
        migrate(store, vectors, "books", ["s1"], lambda: NOW, lambda _: None, -5)
    with pytest.raises(ValueError, match="grace_seconds"):
        resume_move(store, vectors, "books", lambda: NOW, lambda _: None, -5)
    with pytest.raises(ValueError, match="grace_seconds"):
        alias_feature(store, vectors, "books", "libros", lambda _: None, -1)


def test_upsert_reloads_map_when_old_feature_is_unknown_in_cache():
    folder = FakeMapFolder()
    folder.store_for("items").save(_doc())
    client = ShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=lambda: NOW,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )
    # Prime the cache before another writer adds "toys" on s1.
    client._router.map_for("items", None, write=True)

    updated = folder.store_for("items").load()
    updated.features["toys"] = FeatureInfo(shard_keys=["s1"])
    updated.version += 1
    folder.store_for("items").save(updated)

    calls = []

    def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        if method.__name__ == "retrieve":
            return [_Record(1, {"feature": "toys"})]
        return "ok"

    client._router._invoke = invoke
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.upsert(
            "items",
            points=[PointStruct(id=1, vector=[0.1, 0.2], payload={"feature": "books"})],
            wait=False,
        )

    deletes = [flat for name, flat in calls if name == "delete"]
    assert len(deletes) == 1
    assert deletes[0]["shard_key_selector"] == "s1"
    assert deletes[0]["points_selector"] == [1]
    assert any(item.category is FeatureMovedWarning for item in caught)


def test_selective_filter_write_reloads_map_on_unknown_feature():
    folder = FakeMapFolder()
    folder.store_for("items").save(_doc())
    client = ShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=lambda: NOW,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )
    # Prime cache before another client adds "toys" on s1.
    client._router.map_for("items", None, write=True)

    updated = folder.store_for("items").load()
    updated.features["toys"] = FeatureInfo(shard_keys=["s1"])
    updated.version += 1
    folder.store_for("items").save(updated)

    calls = []

    def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        return "ok"

    client._router._invoke = invoke
    flt = Filter(must=[FieldCondition(key="feature", match=MatchValue(value="toys"))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.delete("items", points_selector=flt)

    assert len(calls) == 1
    assert calls[0][1]["shard_key_selector"] == ["s1"]
    assert not any(item.category is FanOutWarning for item in caught)


def test_async_selective_and_upsert_reload_unknown_features():
    asyncio.run(_async_reload_checks())


async def _async_reload_checks():
    folder = FakeMapFolder()
    folder.store_for("items").save(_doc())
    client = AsyncShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=lambda: NOW,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )
    await client._router.map_for("items", None, write=True)

    updated = folder.store_for("items").load()
    updated.features["toys"] = FeatureInfo(shard_keys=["s1"])
    updated.version += 1
    folder.store_for("items").save(updated)

    calls = []

    async def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        if method.__name__ == "retrieve":
            return [_Record(1, {"feature": "toys"})]
        return "ok"

    client._router._invoke = invoke
    flt = Filter(must=[FieldCondition(key="feature", match=MatchValue(value="toys"))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.delete("items", points_selector=flt)
    assert calls[0][1]["shard_key_selector"] == ["s1"]
    assert not any(item.category is FanOutWarning for item in caught)

    # Reset cache to old version, then test upsert move detection reload.
    client._router._caches["items"].put(_doc())
    calls.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.upsert(
            "items",
            points=[PointStruct(id=1, vector=[0.1, 0.2], payload={"feature": "books"})],
            wait=False,
        )
    deletes = [flat for name, flat in calls if name == "delete"]
    assert len(deletes) == 1
    assert deletes[0]["shard_key_selector"] == "s1"
    assert any(item.category is FeatureMovedWarning for item in caught)


class _FakeGrpcNotFound(grpc.RpcError):

    def __init__(self, status: grpc.StatusCode = grpc.StatusCode.NOT_FOUND) -> None:
        super().__init__()
        self._status = status

    def code(self) -> grpc.StatusCode:
        return self._status

    def details(self) -> str:
        return "not found"


class _UnsupportedMoveVectors(FakeVectorStore):

    def check_move_supported(self) -> None:
        raise ShardingError(
            "moves need qdrant-client and server >= 1.17 for insert-only upserts"
        )


def test_migrate_checks_move_support_before_mutating_map():
    store = FakeMapStore()
    store.save(_doc())
    vectors = _UnsupportedMoveVectors()
    with pytest.raises(ShardingError, match="insert-only"):
        migrate(store, vectors, "books", ["s1"], lambda: NOW, lambda _s: None, 0)
    saved = store.load()
    assert saved.version == 1
    assert saved.features["books"].state is FeatureState.stable
    assert saved.features["books"].write_to is None


def test_async_migrate_checks_move_support_before_mutating_map():
    async def run() -> None:
        store = FakeMapStore()
        store.save(_doc())
        vectors = _UnsupportedMoveVectors()

        async def asleep(_seconds: int) -> None:
            return None

        with pytest.raises(ShardingError, match="insert-only"):
            await async_services.migrate(
                store, vectors, "books", ["s1"], lambda: NOW, asleep, 0
            )
        saved = store.load()
        assert saved.version == 1
        assert saved.features["books"].state is FeatureState.stable
        assert saved.features["books"].write_to is None

    asyncio.run(run())


def test_probe_ids_ignores_grpc_not_found_on_sync_and_async_clients():
    folder = FakeMapFolder()
    folder.store_for("items").save(_doc())
    client = ShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=lambda: NOW,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )
    probed: list[list[str]] = []

    def invoke(method, flat):
        if method.__name__ == "retrieve":
            return [_Record(1, {})]
        probed.append(list(flat["shard_key_selector"]))
        if flat["shard_key_selector"] == ["s0"]:
            raise _FakeGrpcNotFound(grpc.StatusCode.NOT_FOUND)
        return "ok"

    client._router._invoke = invoke
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert client.set_payload("items", payload={"tag": "x"}, points=[1]) == "ok"
    assert probed == [["s0"], ["s1"]]

    async def run_async() -> None:
        aclient = AsyncShardedQdrantClient(
            url="http://127.0.0.1:1",
            map_folder=folder,
            now_fn=lambda: NOW,
            prefer_grpc=False,
            timeout=1,
            **compatibility_kwargs(),
        )
        aprobed: list[list[str]] = []

        async def ainvoke(method, flat):
            if method.__name__ == "retrieve":
                return [_Record(1, {})]
            aprobed.append(list(flat["shard_key_selector"]))
            if flat["shard_key_selector"] == ["s0"]:
                raise _FakeGrpcNotFound(grpc.StatusCode.NOT_FOUND)
            return "ok"

        aclient._router._invoke = ainvoke
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert await aclient.set_payload("items", payload={"tag": "x"}, points=[1]) == "ok"
        assert aprobed == [["s0"], ["s1"]]

    asyncio.run(run_async())
