import asyncio
from datetime import datetime, timezone

import pytest

from qdrant_custom_sharding import MoveOrder, RebalancePlan, StopReason
from qdrant_custom_sharding.errors import InvalidMove
from qdrant_custom_sharding.load import snapshot_from_facets
from qdrant_custom_sharding.map_cache import MapCache, Reason
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc, Point
from qdrant_custom_sharding.planning import plan_rebalance
from qdrant_custom_sharding.services import (
    add_feature,
    create_sharded_collection,
    plan_rebalance as plan_from_store,
)
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _doc(
    slots: list[str],
    features: dict[str, FeatureInfo],
    slot_load: dict[str, int] | None = None,
    feature_type: str = "keyword",
    collection: str = "items",
) -> MapDoc:
    loads = {slot: 0 for slot in slots} if slot_load is None else slot_load
    return MapDoc(
        collection=collection,
        feature_field="feature",
        slots=list(slots),
        features=features,
        slot_load=loads,
        feature_type=feature_type,
    )


def _snap(
    slots: list[str],
    facets: dict[str, dict[str, int]],
    collection: str = "items",
):
    return snapshot_from_facets(collection, slots, facets)


def test_balanced_snapshot_returns_no_moves():
    doc = _doc(["s0", "s1"], {"books": FeatureInfo(shard_keys=["s0"])})
    snapshot = _snap(
        ["s0", "s1"],
        {"s0": {"books": 10}, "s1": {"books": 10}},
    )
    plan = plan_rebalance(snapshot, doc)
    assert plan == RebalancePlan("items", (), StopReason.within_ratio)


def test_empty_collection_is_within_ratio():
    doc = _doc(["s0", "s1"], {})
    snapshot = _snap(["s0", "s1"], {"s0": {}, "s1": {}})
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves == ()
    assert plan.reason == StopReason.within_ratio


def test_one_feature_of_ten_points_does_not_move():
    doc = _doc(["s0", "s1", "s2"], {"books": FeatureInfo(shard_keys=["s0"])})
    snapshot = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 10}, "s1": {}, "s2": {}},
    )
    slots = list(doc.slots)
    keys = list(doc.features["books"].shard_keys)
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves == ()
    assert plan.reason == StopReason.no_legal_move
    assert doc.slots == slots
    assert doc.features["books"].shard_keys == keys
    assert doc.version == 1


def test_stacked_features_move_whole_onto_other_slots():
    sizes = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1, "f": 1}
    doc = _doc(
        ["s0", "s1", "s2"],
        {name: FeatureInfo(shard_keys=["s0"]) for name in sizes},
    )
    snapshot = _snap(["s0", "s1", "s2"], {"s0": sizes, "s1": {}, "s2": {}})
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves
    home = {name: "s0" for name in sizes}
    for move in plan.moves:
        assert len(move.to_slots) == 1
        assert move.to_slots[0] != "s0"
        assert move.points == sizes[move.feature]
        home[move.feature] = move.to_slots[0]
    loads = {slot: 0 for slot in ("s0", "s1", "s2")}
    for name, slot in home.items():
        loads[slot] += sizes[name]
    assert max(loads.values()) <= 1.5 * (sum(sizes.values()) / 3)
    assert len(set(home.values())) > 1


def test_moves_onto_one_coldest_slot_and_leaves_the_map():
    doc = _doc(
        ["s0", "s1", "s2"],
        {
            "books": FeatureInfo(shard_keys=["s0"]),
            "tail": FeatureInfo(shard_keys=["s0"]),
        },
    )
    snapshot = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 4, "tail": 6}, "s1": {}, "s2": {}},
    )
    slots = list(doc.slots)
    books = list(doc.features["books"].shard_keys)
    tail = list(doc.features["tail"].shard_keys)
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves == (MoveOrder("books", ("s1",), 4),)
    assert len(plan.moves[0].to_slots) == 1
    assert plan.reason == StopReason.no_legal_move
    assert doc.slots == slots
    assert doc.features["books"].shard_keys == books
    assert doc.features["tail"].shard_keys == tail
    assert doc.version == 1


def test_width_two_on_two_slots_cannot_move():
    doc = _doc(
        ["s0", "s1"],
        {"books": FeatureInfo(shard_keys=["s0", "s1"])},
    )
    snapshot = _snap(["s0", "s1"], {"s0": {"books": 10}, "s1": {}})
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves == ()
    assert plan.reason == StopReason.no_legal_move


def test_stale_slot_load_does_not_change_the_plan():
    features = {
        "books": FeatureInfo(shard_keys=["s0"]),
        "toys": FeatureInfo(shard_keys=["s0"]),
    }
    snapshot = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 8, "toys": 2}, "s1": {}, "s2": {}},
    )
    stale = _doc(
        ["s0", "s1", "s2"],
        features,
        {"s0": 0, "s1": 100, "s2": 0},
    )
    fresh = _doc(
        ["s0", "s1", "s2"],
        features,
        {"s0": 10, "s1": 0, "s2": 0},
    )
    plan = plan_rebalance(snapshot, stale)
    assert plan == plan_rebalance(snapshot, fresh)
    assert plan.moves == (MoveOrder("toys", ("s1",), 2),)


def test_unknown_facet_key_marks_the_slot_to_relieve():
    doc = _doc(
        ["s0", "s1", "s2"],
        {
            "books": FeatureInfo(shard_keys=["s0"]),
            "toys": FeatureInfo(shard_keys=["s1"]),
        },
    )
    quiet = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 2}, "s1": {"toys": 2}, "s2": {}},
    )
    loud = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 2, "ghost": 20}, "s1": {"toys": 2}, "s2": {}},
    )
    assert plan_rebalance(quiet, doc).moves == ()
    assert plan_rebalance(quiet, doc).reason == StopReason.within_ratio
    plan = plan_rebalance(loud, doc)
    assert plan.moves == (MoveOrder("books", ("s2",), 2),)


def test_alias_counts_on_the_current_slots_are_points():
    doc = _doc(
        ["s0", "s1", "s2"],
        {
            "7": FeatureInfo(shard_keys=["s0"], aliases=[8]),
            "toys": FeatureInfo(shard_keys=["s0"]),
        },
        feature_type="integer",
    )
    snapshot = _snap(
        ["s0", "s1", "s2"],
        {
            "s0": {"7": 3, "8": 2, "toys": 6},
            "s1": {"8": 9},
            "s2": {},
        },
    )
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves[0] == MoveOrder("7", ("s2",), 5)


def test_busy_slots_stay_out_of_the_target_list():
    doc = _doc(
        ["s0", "s1", "s2"],
        {
            "books": FeatureInfo(shard_keys=["s0"]),
            "moving": FeatureInfo(
                shard_keys=["s1"],
                state=FeatureState.copying,
                write_to=["s1"],
            ),
        },
    )
    snapshot = _snap(
        ["s0", "s1", "s2"],
        {"s0": {"books": 4, "ghost": 8}, "s1": {}, "s2": {}},
    )
    plan = plan_rebalance(snapshot, doc)
    assert plan.moves == (MoveOrder("books", ("s2",), 4),)


def test_rejects_a_mismatched_collection_or_empty_slots_or_empty_keys():
    doc = _doc(["s0"], {"books": FeatureInfo(shard_keys=["s0"])})
    other = _snap(["s0"], {"s0": {"books": 1}}, collection="other")
    with pytest.raises(InvalidMove):
        plan_rebalance(other, doc)
    empty = _doc([], {})
    blank = _snap([], {})
    with pytest.raises(InvalidMove):
        plan_rebalance(blank, empty)
    broken = _doc(
        ["s0", "s1"],
        {"books": FeatureInfo(shard_keys=[])},
    )
    skewed = _snap(["s0", "s1"], {"s0": {"books": 10}, "s1": {}})
    with pytest.raises(InvalidMove):
        plan_rebalance(skewed, broken)


def test_package_exports_plan_types_and_not_the_function():
    import qdrant_custom_sharding as sdk

    assert sdk.StopReason.within_ratio == "within_ratio"
    assert sdk.StopReason.no_legal_move == "no_legal_move"
    assert "plan_rebalance" not in sdk.__all__


class _AsyncCounts:

    def __init__(self, inner: FakeVectorStore) -> None:
        self._inner = inner

    async def feature_counts(
        self, collection: str, slot: str, field: str
    ) -> dict[str, int]:
        return self._inner.feature_counts(collection, slot, field)


def _ready(
    n_slots: int = 3,
) -> tuple[FakeMapFolder, FakeMapStore, FakeVectorStore]:
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    store = folder.store_for("items")
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=n_slots,
        vectors_config={"size": 2},
    )
    return folder, store, vectors


def _put(vectors: FakeVectorStore, slot: str, point_id: int, value: str) -> None:
    vectors.upsert(
        "items",
        slot,
        [Point(id=point_id, vector=[1.0, 0.0], payload={"feature": value})],
    )


def _admin(folder: FakeMapFolder, vectors: FakeVectorStore) -> tuple[ShardingAdmin, MapCache]:
    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    admin = ShardingAdmin(
        folder,
        lambda _collection: cache,
        lambda: NOW,
        lambda _seconds: None,
        vectors,
    )
    return admin, cache


def test_admin_plan_rebalance_returns_the_plan_and_does_not_save():
    folder, store, vectors = _ready(3)
    add_feature(store, "books")
    for point_id in range(4):
        _put(vectors, "s0", point_id, "books")
    for point_id in range(4, 12):
        _put(vectors, "s0", point_id, "ghost")
    before_version = store.load().version
    before_text = store._text
    expected = RebalancePlan(
        "items",
        (MoveOrder("books", ("s1",), 4),),
        StopReason.no_legal_move,
    )
    assert plan_from_store(store, vectors) == expected
    assert store.load().version == before_version
    assert store._text == before_text

    admin, cache = _admin(folder, vectors)
    assert admin.plan_rebalance("items") == expected
    assert store.load().version == before_version
    assert store._text == before_text
    store.fail_with_unavailable = True
    cached, reason = cache.get()
    assert cached is None
    assert reason is Reason.no_map
    store.fail_with_unavailable = False
    asyncio.run(_async_plan_leaves_the_map(folder, vectors, before_version, expected))


async def _async_plan_leaves_the_map(
    folder: FakeMapFolder,
    vectors: FakeVectorStore,
    before_version: int,
    expected: RebalancePlan,
) -> None:
    from qdrant_custom_sharding.async_services import plan_rebalance as async_plan
    from qdrant_custom_sharding.async_sharded_client import AsyncShardingAdmin

    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    counts = _AsyncCounts(vectors)
    admin = AsyncShardingAdmin(
        folder,
        lambda _collection: cache,
        lambda: NOW,
        _asleep,
        counts,
    )
    store = folder.store_for("items")
    assert await async_plan(store, counts) == expected
    assert await admin.plan_rebalance("items") == expected
    assert store.load().version == before_version
    store.fail_with_unavailable = True
    cached, reason = cache.get()
    assert cached is None
    assert reason is Reason.no_map


async def _asleep(_seconds: int) -> None:
    return None
