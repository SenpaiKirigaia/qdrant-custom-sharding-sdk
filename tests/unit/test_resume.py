from datetime import datetime, timedelta, timezone

from qdrant_custom_sharding.models import FeatureState, Point
from qdrant_custom_sharding.services import (
    add_feature,
    copy_feature,
    create_sharded_collection,
    cutover,
    open_dual,
    resume_move,
    start_move,
)
from tests.fakes import FakeMapStore, FakeVectorStore

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


def _put(vectors: FakeVectorStore, slot: str, point_id: int) -> None:
    vectors.upsert(
        "items",
        slot,
        [Point(id=point_id, vector=[1.0, 0.0], payload={"feature": "books"})],
    )


def _ids(vectors: FakeVectorStore, slot: str) -> list[int | str]:
    found: list[int | str] = []
    for batch in vectors.scroll("items", slot, "feature", ["books"], 10):
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


def test_stable_resume_does_not_change_version():
    store, vectors = _ready()
    add_feature(store, "books")
    version = store.load().version
    now_fn, sleep_fn, slept = _clock()
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    assert store.load().version == version
    assert store.load().features["books"].state == FeatureState.stable
    assert slept == []


def test_second_resume_after_stable_does_not_change_version():
    store, vectors = _ready()
    add_feature(store, "books")
    now_fn, sleep_fn, slept = _clock()
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    version = store.load().version
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    assert store.load().version == version
    assert slept == []


def test_resume_from_copying_sleeps_leftover_then_lands_on_write_to():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    start_move(store, "books", ["s1"], NOW, 10)
    write_to = store.load().features["books"].write_to
    assert write_to == ["s1"]
    target = list(write_to)
    now_fn, sleep_fn, slept = _clock()
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    moved = store.load().features["books"]
    assert slept[0] == 10
    assert slept == [10, 60, 60]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == target
    assert moved.write_to is None
    assert _ids(vectors, "s1") == [1]
    assert _ids(vectors, "s0") == []


def test_resume_from_dual_sleeps_leftover_then_lands_on_write_to():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    start_move(store, "books", ["s1"], NOW, 0)
    copy_feature(store, vectors, "books", NOW)
    open_dual(store, "books", NOW, 15)
    write_to = store.load().features["books"].write_to
    assert write_to == ["s1"]
    target = list(write_to)
    now_fn, sleep_fn, slept = _clock()
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    moved = store.load().features["books"]
    assert slept == [15, 60]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == target
    assert _ids(vectors, "s1") == [1]


def test_resume_from_cutover_sleeps_leftover_then_drops_the_old_slot():
    store, vectors = _ready()
    add_feature(store, "books")
    _put(vectors, "s0", 1)
    start_move(store, "books", ["s1"], NOW, 0)
    copy_feature(store, vectors, "books", NOW)
    open_dual(store, "books", NOW, 0)
    cutover(store, vectors, "books", NOW, 8)
    assert store.load().features["books"].state == FeatureState.cutover
    now_fn, sleep_fn, slept = _clock()
    resume_move(store, vectors, "books", now_fn, sleep_fn, 60)
    moved = store.load().features["books"]
    assert slept == [8]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == ["s1"]
    assert _ids(vectors, "s0") == []
