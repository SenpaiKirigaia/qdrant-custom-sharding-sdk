from datetime import datetime, timedelta, timezone

import pytest

from qdrant_custom_sharding.errors import InvalidMove
from qdrant_custom_sharding.map_cache import MapCache
from qdrant_custom_sharding.models import FeatureState, Point
from qdrant_custom_sharding.planning import MoveOrder, RebalancePlan, StopReason
from qdrant_custom_sharding.services import (
    add_feature,
    apply_rebalance,
    create_sharded_collection,
    start_move,
)
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _ready(n_slots: int = 2) -> tuple[FakeMapStore, FakeVectorStore]:
    store = FakeMapStore()
    vectors = FakeVectorStore()
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=n_slots,
        vectors_config={"size": 2},
    )
    return store, vectors


def _put(vectors: FakeVectorStore, slot: str, point_id: int, value: str = "books") -> None:
    vectors.upsert(
        "items",
        slot,
        [Point(id=point_id, vector=[1.0, 0.0], payload={"feature": value})],
    )


def _ids(vectors: FakeVectorStore, slot: str, value: str = "books") -> list[int | str]:
    found: list[int | str] = []
    for batch in vectors.scroll("items", slot, "feature", [value], 10):
        found.extend(point.id for point in batch)
    return found


def _clock():
    clock = [NOW]
    slept: list[int] = []

    def now_fn() -> datetime:
        return clock[0]

    def sleep_fn(seconds: int) -> None:
        slept.append(seconds)
        clock[0] = clock[0] + timedelta(seconds=seconds)

    return now_fn, sleep_fn, slept


def _plan(collection: str, *moves: MoveOrder) -> RebalancePlan:
    return RebalancePlan(collection, moves, StopReason.within_ratio)


def test_empty_plan_leaves_the_version_unchanged():
    store, vectors = _ready()
    add_feature(store, "books")
    version = store.load().version
    now_fn, sleep_fn, slept = _clock()
    apply_rebalance(store, vectors, _plan("items"), now_fn, sleep_fn, 0)
    assert store.load().version == version
    assert slept == []


def test_stable_feature_moves_onto_the_planned_slot():
    store, vectors = _ready()
    add_feature(store, "books")
    assert store.load().features["books"].shard_keys == ["s0"]
    _put(vectors, "s0", 1)
    now_fn, sleep_fn, _slept = _clock()
    apply_rebalance(
        store,
        vectors,
        _plan("items", MoveOrder("books", ("s1",), 1)),
        now_fn,
        sleep_fn,
        0,
    )
    moved = store.load().features["books"]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == ["s1"]
    assert moved.write_to is None
    assert _ids(vectors, "s1") == [1]
    assert _ids(vectors, "s0") == []


def test_planned_slots_already_held_do_not_bump_version():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    version = store.load().version
    now_fn, sleep_fn, slept = _clock()
    apply_rebalance(
        store,
        vectors,
        _plan("items", MoveOrder("books", ("s0",), 1)),
        now_fn,
        sleep_fn,
        0,
    )
    stayed = store.load().features["books"]
    assert store.load().version == version
    assert stayed.state == FeatureState.stable
    assert stayed.shard_keys == ["s0"]
    assert _ids(vectors, "s0") == [1]
    assert slept == []


def test_copying_onto_the_planned_slots_finishes_stable():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    start_move(store, "books", ["s1"], NOW, 0)
    copying = store.load().features["books"]
    assert copying.state == FeatureState.copying
    assert copying.write_to == ["s1"]
    now_fn, sleep_fn, _slept = _clock()
    apply_rebalance(
        store,
        vectors,
        _plan("items", MoveOrder("books", ("s1",), 1)),
        now_fn,
        sleep_fn,
        0,
    )
    moved = store.load().features["books"]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == ["s1"]
    assert moved.write_to is None
    assert _ids(vectors, "s1") == [1]
    assert _ids(vectors, "s0") == []


def test_copying_onto_other_slots_raises_and_leaves_the_later_feature():
    store, vectors = _ready(4)
    add_feature(store, "books")
    add_feature(store, "films")
    assert store.load().features["books"].shard_keys == ["s0"]
    assert store.load().features["films"].shard_keys == ["s1"]
    start_move(store, "books", ["s2"], NOW, 0)
    version = store.load().version
    now_fn, sleep_fn, _slept = _clock()
    with pytest.raises(InvalidMove):
        apply_rebalance(
            store,
            vectors,
            _plan(
                "items",
                MoveOrder("books", ("s3",), 1),
                MoveOrder("films", ("s3",), 1),
            ),
            now_fn,
            sleep_fn,
            0,
        )
    books = store.load().features["books"]
    films = store.load().features["films"]
    assert store.load().version == version
    assert books.state == FeatureState.copying
    assert books.write_to == ["s2"]
    assert books.shard_keys == ["s0"]
    assert films.state == FeatureState.stable
    assert films.shard_keys == ["s1"]


def test_other_collection_raises_and_does_not_bump_version():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    version = store.load().version
    now_fn, sleep_fn, _slept = _clock()
    with pytest.raises(InvalidMove):
        apply_rebalance(
            store,
            vectors,
            _plan("other", MoveOrder("books", ("s1",), 1)),
            now_fn,
            sleep_fn,
            0,
        )
    stayed = store.load().features["books"]
    assert store.load().version == version
    assert stayed.state == FeatureState.stable
    assert stayed.shard_keys == ["s0"]
    assert _ids(vectors, "s0") == [1]


def test_unknown_feature_raises_and_leaves_the_later_feature():
    store, vectors = _ready(3)
    add_feature(store, "books")
    version = store.load().version
    now_fn, sleep_fn, _slept = _clock()
    with pytest.raises(InvalidMove):
        apply_rebalance(
            store,
            vectors,
            _plan(
                "items",
                MoveOrder("missing", ("s1",), 1),
                MoveOrder("books", ("s2",), 1),
            ),
            now_fn,
            sleep_fn,
            0,
        )
    stayed = store.load().features["books"]
    assert store.load().version == version
    assert stayed.state == FeatureState.stable
    assert stayed.shard_keys == ["s0"]


def test_apply_rebalance_is_not_a_package_export():
    import qdrant_custom_sharding as sdk

    assert "apply_rebalance" not in sdk.__all__


def test_admin_empty_plan_does_not_bump_the_cached_version():
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    store = folder.store_for("items")
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=2,
        vectors_config={"size": 2},
    )
    add_feature(store, "books")
    version = store.load().version
    cache = MapCache(store, 30, lambda: NOW)
    cache.put(store.load())
    now_fn, sleep_fn, slept = _clock()
    admin = ShardingAdmin(folder, lambda _name: cache, now_fn, sleep_fn, vectors)
    doc = admin.apply_rebalance("items", _plan("items"), 0)
    seen, reason = cache.get()
    assert doc.version == version
    assert store.load().version == version
    assert reason is None
    assert seen is not None
    assert seen.version == version
    assert slept == []


def test_admin_apply_puts_the_moved_map_in_the_cache():
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    store = folder.store_for("items")
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=2,
        vectors_config={"size": 2},
    )
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    cache = MapCache(store, 30, lambda: NOW)
    cache.put(store.load())
    now_fn, sleep_fn, _slept = _clock()
    admin = ShardingAdmin(folder, lambda _name: cache, now_fn, sleep_fn, vectors)
    doc = admin.apply_rebalance("items", _plan("items", MoveOrder("books", ("s1",), 1)), 0)
    seen, reason = cache.get()
    assert reason is None
    assert seen is not None
    assert doc.features["books"].shard_keys == ["s1"]
    assert seen.features["books"].shard_keys == ["s1"]
    assert seen.version == doc.version
    assert _ids(vectors, "s0") == []
