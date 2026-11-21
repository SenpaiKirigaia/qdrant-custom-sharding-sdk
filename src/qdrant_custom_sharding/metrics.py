from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_custom_sharding.stats import CollectionStats


@dataclass(frozen=True)
class CollectionRoutingMetrics:
    collection: str
    routed_reads: int = 0
    fanout_reads: int = 0
    routed_writes: int = 0
    fanout_writes: int = 0
    feature_moves: int = 0
    write_denied: int = 0
    cache_hits: int = 0
    cache_reloads: int = 0
    cache_stale: int = 0
    cache_unavailable: int = 0
    methods: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "routed_reads": self.routed_reads,
            "fanout_reads": self.fanout_reads,
            "routed_writes": self.routed_writes,
            "fanout_writes": self.fanout_writes,
            "feature_moves": self.feature_moves,
            "write_denied": self.write_denied,
            "cache_hits": self.cache_hits,
            "cache_reloads": self.cache_reloads,
            "cache_stale": self.cache_stale,
            "cache_unavailable": self.cache_unavailable,
            "methods": dict(self.methods or {}),
        }


@dataclass(frozen=True)
class RoutingMetricsSnapshot:
    collections: dict[str, CollectionRoutingMetrics]
    total_routed_reads: int
    total_fanout_reads: int
    total_routed_writes: int
    total_fanout_writes: int
    total_feature_moves: int
    total_write_denied: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "collections": {
                name: row.to_dict() for name, row in self.collections.items()
            },
            "total_routed_reads": self.total_routed_reads,
            "total_fanout_reads": self.total_fanout_reads,
            "total_routed_writes": self.total_routed_writes,
            "total_fanout_writes": self.total_fanout_writes,
            "total_feature_moves": self.total_feature_moves,
            "total_write_denied": self.total_write_denied,
        }


class MetricsRecorder(Protocol):

    def record_read(
        self, collection: str, method: str, fanout: bool, slot_count: int
    ) -> None: ...

    def record_write(
        self, collection: str, method: str, fanout: bool, slot_count: int
    ) -> None: ...

    def record_feature_move(self, collection: str, count: int) -> None: ...

    def record_cache_event(self, collection: str, outcome: str) -> None: ...

    def record_write_denied(self, collection: str) -> None: ...

    def snapshot(self) -> RoutingMetricsSnapshot: ...


class _MutableCollectionCounters:
    def __init__(self) -> None:
        self.routed_reads = 0
        self.fanout_reads = 0
        self.routed_writes = 0
        self.fanout_writes = 0
        self.feature_moves = 0
        self.write_denied = 0
        self.cache_hits = 0
        self.cache_reloads = 0
        self.cache_stale = 0
        self.cache_unavailable = 0
        self.methods: dict[str, int] = {}


class InMemoryMetrics:
    """Thread-safe in-process routing metrics recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_collection: dict[str, _MutableCollectionCounters] = {}

    def record_read(
        self, collection: str, method: str, fanout: bool, slot_count: int
    ) -> None:
        with self._lock:
            row = self._row(collection)
            if fanout:
                row.fanout_reads += 1
            else:
                row.routed_reads += 1
            row.methods[method] = row.methods.get(method, 0) + 1

    def record_write(
        self, collection: str, method: str, fanout: bool, slot_count: int
    ) -> None:
        with self._lock:
            row = self._row(collection)
            if fanout:
                row.fanout_writes += 1
            else:
                row.routed_writes += 1
            row.methods[method] = row.methods.get(method, 0) + 1

    def record_feature_move(self, collection: str, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._row(collection).feature_moves += count

    def record_cache_event(self, collection: str, outcome: str) -> None:
        with self._lock:
            row = self._row(collection)
            if outcome == "hit":
                row.cache_hits += 1
            elif outcome == "reload":
                row.cache_reloads += 1
            elif outcome == "stale":
                row.cache_stale += 1
            elif outcome == "unavailable":
                row.cache_unavailable += 1

    def record_write_denied(self, collection: str) -> None:
        with self._lock:
            self._row(collection).write_denied += 1

    def reset(self) -> None:
        with self._lock:
            self._by_collection.clear()

    def snapshot(self) -> RoutingMetricsSnapshot:
        with self._lock:
            collections = {
                name: CollectionRoutingMetrics(
                    collection=name,
                    routed_reads=row.routed_reads,
                    fanout_reads=row.fanout_reads,
                    routed_writes=row.routed_writes,
                    fanout_writes=row.fanout_writes,
                    feature_moves=row.feature_moves,
                    write_denied=row.write_denied,
                    cache_hits=row.cache_hits,
                    cache_reloads=row.cache_reloads,
                    cache_stale=row.cache_stale,
                    cache_unavailable=row.cache_unavailable,
                    methods=dict(row.methods),
                )
                for name, row in sorted(self._by_collection.items())
            }
        return RoutingMetricsSnapshot(
            collections=collections,
            total_routed_reads=sum(r.routed_reads for r in collections.values()),
            total_fanout_reads=sum(r.fanout_reads for r in collections.values()),
            total_routed_writes=sum(r.routed_writes for r in collections.values()),
            total_fanout_writes=sum(r.fanout_writes for r in collections.values()),
            total_feature_moves=sum(r.feature_moves for r in collections.values()),
            total_write_denied=sum(r.write_denied for r in collections.values()),
        )

    def _row(self, collection: str) -> _MutableCollectionCounters:
        row = self._by_collection.get(collection)
        if row is None:
            row = _MutableCollectionCounters()
            self._by_collection[collection] = row
        return row


def render_prometheus(
    collections: Sequence[CollectionStats],
    routing: RoutingMetricsSnapshot | None = None,
    top_features: int = 50,
) -> str:
    """Format collection statistics and routing counters in Prometheus text format."""
    lines: list[str] = []
    if collections:
        lines.extend(
            [
                "# HELP qdrant_sharding_collection_points_total Total points in a sharded collection.",
                "# TYPE qdrant_sharding_collection_points_total gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_points_total{{{label}}} {item.total_points}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_slots Number of slots in a sharded collection.",
                "# TYPE qdrant_sharding_collection_slots gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_slots{{{label}}} {item.slot_count}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_features Number of tracked features in a collection map.",
                "# TYPE qdrant_sharding_collection_features gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_features{{{label}}} {item.feature_count}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_imbalance_ratio Max slot load divided by mean slot load.",
                "# TYPE qdrant_sharding_collection_imbalance_ratio gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_imbalance_ratio{{{label}}} {item.imbalance_ratio:g}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_gini Gini coefficient of slot load distribution.",
                "# TYPE qdrant_sharding_collection_gini gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_gini{{{label}}} {item.gini_coefficient:g}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_active_moves Features currently in a non-stable move state.",
                "# TYPE qdrant_sharding_collection_active_moves gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_active_moves{{{label}}} {len(item.active_moves)}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_collection_map_version Current map version number.",
                "# TYPE qdrant_sharding_collection_map_version gauge",
            ]
        )
        for item in collections:
            label = _labels(collection=item.collection)
            lines.append(
                f"qdrant_sharding_collection_map_version{{{label}}} {item.version}"
            )

        lines.extend(
            [
                "# HELP qdrant_sharding_slot_points Point count stored on a slot.",
                "# TYPE qdrant_sharding_slot_points gauge",
            ]
        )
        for item in collections:
            for slot in item.slots:
                label = _labels(collection=item.collection, slot=slot.slot)
                lines.append(f"qdrant_sharding_slot_points{{{label}}} {slot.points}")

        lines.extend(
            [
                "# HELP qdrant_sharding_slot_features Number of features assigned to a slot.",
                "# TYPE qdrant_sharding_slot_features gauge",
            ]
        )
        for item in collections:
            for slot in item.slots:
                label = _labels(collection=item.collection, slot=slot.slot)
                lines.append(
                    f"qdrant_sharding_slot_features{{{label}}} {slot.feature_count}"
                )

        lines.extend(
            [
                "# HELP qdrant_sharding_slot_busy 1 if the slot participates in an active move, else 0.",
                "# TYPE qdrant_sharding_slot_busy gauge",
            ]
        )
        for item in collections:
            for slot in item.slots:
                label = _labels(collection=item.collection, slot=slot.slot)
                busy = 1 if slot.busy else 0
                lines.append(f"qdrant_sharding_slot_busy{{{label}}} {busy}")

        if top_features > 0:
            lines.extend(
                [
                    "# HELP qdrant_sharding_feature_points Point count for top features by size.",
                    "# TYPE qdrant_sharding_feature_points gauge",
                ]
            )
            for item in collections:
                for feature in item.features[:top_features]:
                    label = _labels(
                        collection=item.collection,
                        feature=feature.name,
                        state=feature.state.value,
                    )
                    lines.append(
                        f"qdrant_sharding_feature_points{{{label}}} {feature.points}"
                    )

    if routing is not None and routing.collections:
        metric_specs = [
            (
                "qdrant_sharding_routed_reads_total",
                "Reads routed to specific shard keys.",
                "routed_reads",
            ),
            (
                "qdrant_sharding_fanout_reads_total",
                "Reads that fanned out across all shards.",
                "fanout_reads",
            ),
            (
                "qdrant_sharding_routed_writes_total",
                "Writes routed to specific shard keys.",
                "routed_writes",
            ),
            (
                "qdrant_sharding_fanout_writes_total",
                "Writes that fanned out across slots.",
                "fanout_writes",
            ),
            (
                "qdrant_sharding_feature_moves_total",
                "Points moved between feature slots during updates.",
                "feature_moves",
            ),
            (
                "qdrant_sharding_write_denied_total",
                "Writes denied because the map store was unavailable.",
                "write_denied",
            ),
            (
                "qdrant_sharding_cache_stale_total",
                "Operations served from a stale cached map.",
                "cache_stale",
            ),
            (
                "qdrant_sharding_cache_unavailable_total",
                "Map loads that failed with MapUnavailable and no cached map.",
                "cache_unavailable",
            ),
        ]
        for metric_name, help_text, attr in metric_specs:
            lines.append(f"# HELP {metric_name} {help_text}")
            lines.append(f"# TYPE {metric_name} counter")
            for name, row in routing.collections.items():
                label = _labels(collection=name)
                lines.append(f"{metric_name}{{{label}}} {getattr(row, attr)}")

    return "\n".join(lines) + ("\n" if lines else "")


def _labels(**pairs: str) -> str:
    return ",".join(
        f'{key}="{_escape(value)}"' for key, value in pairs.items()
    )


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
