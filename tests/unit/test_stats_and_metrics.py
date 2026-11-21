import asyncio
import warnings
from datetime import datetime, timedelta, timezone

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from qdrant_custom_sharding.async_sharded_client import AsyncShardedQdrantClient
from qdrant_custom_sharding.load import snapshot_from_facets
from qdrant_custom_sharding.metrics import (
    InMemoryMetrics,
    render_prometheus,
)
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc
from qdrant_custom_sharding.sharded_client import ShardedQdrantClient
from qdrant_custom_sharding.stats import collection_stats, recommend_splits
from tests.fakes import FakeMapFolder
from tests.local_qdrant import compatibility_kwargs

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _skewed_doc() -> MapDoc:
    return MapDoc(
        collection="catalog",
        feature_field="category",
        feature_type="keyword",
        slots=["s0", "s1", "s2", "s3"],
        features={
            "whale": FeatureInfo(shard_keys=["s0"], points_count=900),
            "medium": FeatureInfo(shard_keys=["s1"], points_count=80),
            "small": FeatureInfo(
                shard_keys=["s2"],
                state=FeatureState.copying,
                write_to=["s3"],
                points_count=20,
                grace_until=NOW + timedelta(seconds=45),
                copied=False,
            ),
        },
        slot_load={"s0": 900, "s1": 80, "s2": 20, "s3": 0},
        version=5,
    )


def test_collection_stats_computes_skew_gini_histogram_and_moves_from_map():
    doc = _skewed_doc()
    stats = collection_stats(doc, now=NOW)
    assert stats.collection == "catalog"
    assert stats.source == "map"
    assert stats.version == 5
    assert stats.slot_count == 4
    assert stats.feature_count == 3
    assert stats.total_points == 1000
    assert stats.orphan_points == 0
    assert stats.mean_slot_load == 250.0
    assert stats.max_slot_load == 900
    assert stats.min_slot_load == 0
    assert stats.imbalance_ratio == pytest.approx(3.6)
    assert stats.needs_rebalance is True
    assert 0.5 < stats.gini_coefficient < 1.0

    # Active move is captured with remaining grace seconds.
    assert len(stats.active_moves) == 1
    assert stats.active_moves[0].name == "small"
    assert stats.active_moves[0].state is FeatureState.copying
    assert stats.active_moves[0].remaining_grace_seconds == 45

    # Slot statuses reflect hot / balanced / busy / empty.
    by_slot = {row.slot: row for row in stats.slots}
    assert by_slot["s0"].status == "hot"
    assert by_slot["s1"].status == "balanced"
    assert by_slot["s2"].busy is True
    assert by_slot["s2"].status == "busy"
    assert by_slot["s3"].busy is True

    # Features are sorted descending by point count.
    assert [row.name for row in stats.features] == ["whale", "medium", "small"]

    # Histogram buckets sum to total features and total points.
    assert sum(bucket.feature_count for bucket in stats.size_histogram) == 3
    assert sum(bucket.total_points for bucket in stats.size_histogram) == 1000

    # JSON dictionary output is complete.
    payload = stats.to_dict()
    assert payload["collection"] == "catalog"
    assert payload["imbalance_ratio"] == pytest.approx(3.6)
    assert len(payload["slots"]) == 4


def test_collection_stats_uses_live_snapshot_and_counts_orphans():
    doc = MapDoc(
        collection="catalog",
        feature_field="category",
        slots=["s0", "s1"],
        features={
            "books": FeatureInfo(shard_keys=["s0"], aliases=["old_books"], points_count=1),
            "toys": FeatureInfo(shard_keys=["s1"], points_count=1),
        },
        slot_load={"s0": 1, "s1": 1},
    )
    snap = snapshot_from_facets(
        "catalog",
        ["s0", "s1"],
        {"s0": {"books": 60, "old_books": 15, "ghost": 5}, "s1": {"toys": 20}},
    )
    stats = collection_stats(doc, snapshot=snap, now=NOW)
    assert stats.source == "live"
    assert stats.total_points == 100
    assert stats.orphan_points == 5
    by_feature = {row.name: row for row in stats.features}
    assert by_feature["books"].points == 75
    assert by_feature["books"].per_slot_points == {"s0": 75}
    assert by_feature["toys"].points == 20


def test_recommend_splits_identifies_whale_feature_exceeding_slot_budget():
    doc = MapDoc(
        collection="catalog",
        feature_field="category",
        slots=["s0", "s1", "s2", "s3"],
        features={
            "whale": FeatureInfo(shard_keys=["s0"], points_count=900),
            "small": FeatureInfo(shard_keys=["s1"], points_count=100),
        },
        slot_load={"s0": 900, "s1": 100, "s2": 0, "s3": 0},
    )
    recs = recommend_splits(doc)
    assert len(recs) == 1
    assert recs[0].feature == "whale"
    assert recs[0].current_width == 1
    assert recs[0].recommended_partitions == 3


def test_empty_collection_stats_are_zero_safe():
    doc = MapDoc(
        collection="empty",
        feature_field="category",
        slots=["s0", "s1"],
        features={},
        slot_load={"s0": 0, "s1": 0},
    )
    stats = collection_stats(doc, now=NOW)
    assert stats.total_points == 0
    assert stats.imbalance_ratio == 1.0
    assert stats.gini_coefficient == 0.0
    assert stats.needs_rebalance is False


def test_prometheus_exposition_contains_collection_slot_and_routing_metrics():
    stats = collection_stats(_skewed_doc(), now=NOW)
    metrics = InMemoryMetrics()
    metrics.record_read("catalog", "query_points", fanout=False, slot_count=1)
    metrics.record_read("catalog", "count", fanout=True, slot_count=4)
    metrics.record_write("catalog", "upsert", fanout=False, slot_count=1)
    metrics.record_feature_move("catalog", 3)
    metrics.record_cache_event("catalog", "hit")
    metrics.record_cache_event("catalog", "stale")

    text = render_prometheus([stats], metrics.snapshot())
    assert '# TYPE qdrant_sharding_collection_points_total gauge' in text
    assert 'qdrant_sharding_collection_points_total{collection="catalog"} 1000' in text
    assert 'qdrant_sharding_collection_imbalance_ratio{collection="catalog"} 3.6' in text
    assert 'qdrant_sharding_slot_points{collection="catalog",slot="s0"} 900' in text
    assert 'qdrant_sharding_slot_busy{collection="catalog",slot="s2"} 1' in text
    assert 'qdrant_sharding_routed_reads_total{collection="catalog"} 1' in text
    assert 'qdrant_sharding_fanout_reads_total{collection="catalog"} 1' in text
    assert 'qdrant_sharding_routed_writes_total{collection="catalog"} 1' in text
    assert 'qdrant_sharding_feature_moves_total{collection="catalog"} 3' in text
    assert 'qdrant_sharding_cache_stale_total{collection="catalog"} 1' in text


def test_sync_and_async_clients_record_routing_metrics():
    folder = FakeMapFolder()
    folder.store_for("items").save(
        MapDoc(
            collection="items",
            feature_field="feature",
            slots=["s0", "s1"],
            features={"books": FeatureInfo(shard_keys=["s0"], points_count=10)},
            slot_load={"s0": 10, "s1": 0},
        )
    )
    metrics = InMemoryMetrics()
    client = ShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=lambda: NOW,
        metrics=metrics,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )

    def invoke(method, flat):
        if method.__name__ == "retrieve":
            return []
        return "ok"

    client._router._invoke = invoke
    flt = Filter(must=[FieldCondition(key="feature", match=MatchValue(value="books"))])
    client.count("items", count_filter=flt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.count("items")
    client.upsert(
        "items",
        points=[PointStruct(id=1, vector=[0.1], payload={"feature": "books"})],
    )

    snap = client.sharding.metrics.snapshot()
    row = snap.collections["items"]
    assert row.routed_reads == 1
    assert row.fanout_reads == 1
    assert row.routed_writes == 1
    assert client.sharding.stats("items").total_points == 10

    async def _run_async():
        async_metrics = InMemoryMetrics()
        aclient = AsyncShardedQdrantClient(
            url="http://127.0.0.1:1",
            map_folder=folder,
            now_fn=lambda: NOW,
            metrics=async_metrics,
            prefer_grpc=False,
            timeout=1,
            **compatibility_kwargs(),
        )

        async def ainvoke(method, flat):
            if method.__name__ == "retrieve":
                return []
            return "ok"

        aclient._router._invoke = ainvoke
        await aclient.count("items", count_filter=flt)
        stats = await aclient.sharding.stats("items")
        return aclient.sharding.metrics.snapshot(), stats

    asnap, astats = asyncio.run(_run_async())
    assert asnap.collections["items"].routed_reads == 1
    assert astats.total_points == 10
