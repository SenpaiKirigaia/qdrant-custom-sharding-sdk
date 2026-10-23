import pytest

from qdrant_custom_sharding.errors import MapNotFound, MapUnavailable, VersionConflict
from qdrant_custom_sharding.models import FeatureInfo, MapDoc, Point
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore


def _doc(version: int = 1) -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
        version=version,
    )


def test_fake_map_store_keeps_json_and_enforces_version():
    store = FakeMapStore()
    with pytest.raises(MapNotFound):
        store.load()
    with pytest.raises(VersionConflict):
        store.save(_doc(version=2))

    store.save(_doc())
    loaded = store.load()
    assert loaded == _doc()
    loaded.version = 99
    assert store.load().version == 1

    updated = store.load()
    updated.version = 2
    store.save(updated)
    with pytest.raises(VersionConflict):
        store.save(updated)

    store.fail_with_unavailable = True
    with pytest.raises(MapUnavailable):
        store.load()

    store.fail_with_unavailable = False
    store.delete()
    with pytest.raises(MapNotFound):
        store.load()


def test_fake_map_folder_keeps_one_store_per_collection():
    folder = FakeMapFolder()
    books = folder.store_for("books")
    assert folder.store_for("books") is books
    toys = folder.store_for("toys")
    assert toys is not books
    books.save(_doc())
    assert books.load() == _doc()
    with pytest.raises(MapNotFound):
        toys.load()


def test_fake_vector_store_upserts_and_counts_int_and_str():
    store = FakeVectorStore()
    store.create_collection("items", vectors_config={"size": 2}, on_disk_payload=True)
    store.create_feature_index("items", "feature", "keyword")
    store.create_feature_index("items", "subject_id", "integer")
    books = Point(id=1, vector=[1.0, 0.0], payload={"feature": "books"})
    with pytest.raises(KeyError):
        store.upsert("items", "s0", [books])

    store.create_slot("items", "s0")
    store.create_slot("items", "s1")
    named = Point(id=4, vector={"text": [1.0, 0.0]}, payload={"feature": "books"})
    store.upsert(
        "items",
        "s0",
        [
            books,
            Point(id=2, vector=[0.0, 1.0], payload={"feature": "books"}),
            Point(id=3, vector=[1.0, 0.0], payload={"subject_id": 7}),
        ],
    )
    store.upsert("items", "s1", [named])
    named.payload["feature"] = "changed"
    assert store.feature_counts("items", "s0", "feature") == {"books": 2}
    assert store.feature_counts("items", "s0", "subject_id") == {"7": 1}
    stored = next(store.scroll("items", "s1", "feature", ["books"], 10))
    assert stored[0].vector == {"text": [1.0, 0.0]}
    assert stored[0].payload == {"feature": "books"}


def test_fake_insert_missing_skips_existing_ids_and_delete_ids_removes_them():
    store = FakeVectorStore()
    store.create_collection("items")
    store.create_slot("items", "s0")
    store.upsert(
        "items",
        "s0",
        [Point(id=1, vector=[1.0], payload={"feature": "books", "note": "old"})],
    )
    store.insert_missing(
        "items",
        "s0",
        [
            Point(id=1, vector=[9.0], payload={"feature": "books", "note": "new"}),
            Point(id=2, vector=[2.0], payload={"feature": "books", "note": "new"}),
        ],
    )
    assert store.ids_on_slot("items", "s0", [1, 2, 3]) == {1, 2}
    stored = next(store.scroll("items", "s0", "feature", ["books"], 10))
    by_id = {point.id: point for point in stored}
    assert by_id[1].payload["note"] == "old"
    assert by_id[1].vector == [1.0]
    assert by_id[2].payload["note"] == "new"
    store.delete_ids("items", "s0", [1])
    assert store.ids_on_slot("items", "s0", [1, 2]) == {2}


def test_fake_vector_store_scrolls_in_batches_and_deletes_one_feature():
    store = FakeVectorStore()
    store.create_collection("items", vectors_config={"size": 2})
    store.create_slot("items", "s0")
    store.create_slot("items", "s1")
    store.upsert(
        "items",
        "s0",
        [
            Point(id=1, vector=[1.0, 0.0], payload={"feature": "books"}),
            Point(id=2, vector=[0.0, 1.0], payload={"feature": "books"}),
            Point(id=3, vector=[1.0, 0.0], payload={"feature": "toys"}),
        ],
    )
    store.upsert(
        "items",
        "s1",
        [Point(id=4, vector=[1.0, 0.0], payload={"feature": "books"})],
    )

    batches = list(store.scroll("items", "s0", "feature", ["books"], 1))
    assert [[point.id for point in batch] for batch in batches] == [[1], [2]]
    assert batches[0][0].vector == [1.0, 0.0]
    assert list(store.scroll("items", "s0", "feature", ["missing"], 2)) == []

    store.upsert(
        "items",
        "s0",
        [Point(id=5, vector=[1.0, 0.0], payload={"feature": "1"})],
    )
    assert store.feature_counts("items", "s0", "feature") == {
        "books": 2,
        "toys": 1,
        "1": 1,
    }
    store.delete_feature("items", "s0", "feature", ["books", "1"])
    assert store.feature_counts("items", "s0", "feature") == {"toys": 1}
    assert store.feature_counts("items", "s1", "feature") == {"books": 1}


def test_package_exports_errors_state_and_warnings():
    import qdrant_custom_sharding as sdk
    from qdrant_client import QdrantClient

    assert sdk.FeatureState.stable == "stable"
    assert issubclass(sdk.ShardedQdrantClient, QdrantClient)
    assert issubclass(sdk.MoveNotReady, sdk.ShardingError)
    assert issubclass(sdk.WriteDenied, sdk.ShardingError)
    assert issubclass(sdk.FanOutWarning, sdk.ShardingWarning)
    assert issubclass(sdk.FeatureMovedWarning, sdk.ShardingWarning)
    assert issubclass(sdk.StaleMapWarning, sdk.ShardingWarning)
    assert issubclass(sdk.ShardingWarning, UserWarning)
