from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from qdrant_custom_sharding.load import LoadSnapshot
from qdrant_custom_sharding.models import (
    FeatureInfo,
    FeatureState,
    MapDoc,
    remaining_grace_seconds,
)
from qdrant_custom_sharding.planning import BALANCE_RATIO

_HISTOGRAM_BOUNDS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1-10", 1, 10),
    ("11-100", 11, 100),
    ("101-1k", 101, 1_000),
    ("1k-10k", 1_001, 10_000),
    ("10k-100k", 10_001, 100_000),
    ("100k+", 100_001, None),
)


@dataclass(frozen=True)
class SlotStats:
    slot: str
    points: int
    feature_count: int
    share_pct: float
    busy: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "points": self.points,
            "feature_count": self.feature_count,
            "share_pct": self.share_pct,
            "busy": self.busy,
            "status": self.status,
        }


@dataclass(frozen=True)
class FeatureStats:
    name: str
    points: int
    share_pct: float
    shard_keys: tuple[str, ...]
    width: int
    state: FeatureState
    write_to: tuple[str, ...] | None
    aliases: tuple[str, ...]
    copied: bool
    grace_until: str | None
    remaining_grace_seconds: int
    per_slot_points: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": self.points,
            "share_pct": self.share_pct,
            "shard_keys": list(self.shard_keys),
            "width": self.width,
            "state": self.state.value,
            "write_to": None if self.write_to is None else list(self.write_to),
            "aliases": list(self.aliases),
            "copied": self.copied,
            "grace_until": self.grace_until,
            "remaining_grace_seconds": self.remaining_grace_seconds,
            "per_slot_points": dict(self.per_slot_points),
        }


@dataclass(frozen=True)
class HistogramBucket:
    label: str
    min_points: int
    max_points: int | None
    feature_count: int
    total_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "min_points": self.min_points,
            "max_points": self.max_points,
            "feature_count": self.feature_count,
            "total_points": self.total_points,
        }


@dataclass(frozen=True)
class SplitRecommendation:
    feature: str
    current_width: int
    recommended_partitions: int
    points: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "current_width": self.current_width,
            "recommended_partitions": self.recommended_partitions,
            "points": self.points,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CollectionStats:
    collection: str
    feature_field: str
    feature_type: str
    version: int
    source: Literal["map", "live"]
    slot_count: int
    feature_count: int
    total_points: int
    orphan_points: int
    mean_slot_load: float
    median_slot_load: float
    min_slot_load: int
    max_slot_load: int
    p95_slot_load: float
    std_slot_load: float
    imbalance_ratio: float
    gini_coefficient: float
    balance_threshold: float
    needs_rebalance: bool
    active_moves: tuple[FeatureStats, ...]
    slots: tuple[SlotStats, ...]
    features: tuple[FeatureStats, ...]
    size_histogram: tuple[HistogramBucket, ...]
    split_recommendations: tuple[SplitRecommendation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "feature_field": self.feature_field,
            "feature_type": self.feature_type,
            "version": self.version,
            "source": self.source,
            "slot_count": self.slot_count,
            "feature_count": self.feature_count,
            "total_points": self.total_points,
            "orphan_points": self.orphan_points,
            "mean_slot_load": self.mean_slot_load,
            "median_slot_load": self.median_slot_load,
            "min_slot_load": self.min_slot_load,
            "max_slot_load": self.max_slot_load,
            "p95_slot_load": self.p95_slot_load,
            "std_slot_load": self.std_slot_load,
            "imbalance_ratio": self.imbalance_ratio,
            "gini_coefficient": self.gini_coefficient,
            "balance_threshold": self.balance_threshold,
            "needs_rebalance": self.needs_rebalance,
            "active_moves": [item.to_dict() for item in self.active_moves],
            "slots": [item.to_dict() for item in self.slots],
            "features": [item.to_dict() for item in self.features],
            "size_histogram": [item.to_dict() for item in self.size_histogram],
            "split_recommendations": [
                item.to_dict() for item in self.split_recommendations
            ],
        }


def collection_stats(
    doc: MapDoc,
    snapshot: LoadSnapshot | None = None,
    now: datetime | None = None,
    balance_threshold: float = BALANCE_RATIO,
) -> CollectionStats:
    """Compute slot and feature size distributions from a map and optional live counts."""
    current_time = now if now is not None else datetime.now(timezone.utc)
    source: Literal["map", "live"] = "live" if snapshot is not None else "map"
    slot_points, feature_totals, per_slot, orphan_points = _resolve_counts(doc, snapshot)

    loads = [slot_points[slot] for slot in doc.slots]
    total_points = sum(loads)
    slot_count = len(doc.slots)
    mean_load = (total_points / slot_count) if slot_count else 0.0
    median_load = float(statistics.median(loads)) if loads else 0.0
    min_load = min(loads) if loads else 0
    max_load = max(loads) if loads else 0
    p95_load = _percentile(loads, 95.0)
    std_load = float(statistics.pstdev(loads)) if len(loads) > 1 else 0.0
    imbalance = (max_load / mean_load) if mean_load > 0 else 1.0
    gini = _gini(loads)
    needs_rebalance = total_points > 0 and max_load > balance_threshold * mean_load

    busy = doc.busy_slots()
    occupancy = doc.occupancy()
    slot_rows = tuple(
        SlotStats(
            slot=slot,
            points=slot_points[slot],
            feature_count=occupancy.get(slot, 0),
            share_pct=_share(slot_points[slot], total_points),
            busy=slot in busy,
            status=_slot_status(
                slot_points[slot], slot in busy, mean_load, balance_threshold
            ),
        )
        for slot in doc.slots
    )

    feature_rows = tuple(
        sorted(
            (
                _feature_row(
                    name,
                    info,
                    feature_totals[name],
                    per_slot[name],
                    total_points,
                    current_time,
                )
                for name, info in doc.features.items()
            ),
            key=lambda row: (-row.points, row.name),
        )
    )
    active_moves = tuple(
        row for row in feature_rows if row.state != FeatureState.stable
    )
    histogram = _build_histogram(feature_rows)
    splits = tuple(
        _recommend_splits(
            doc, feature_totals, mean_load, busy, balance_threshold
        )
    )

    return CollectionStats(
        collection=doc.collection,
        feature_field=doc.feature_field,
        feature_type=doc.feature_type,
        version=doc.version,
        source=source,
        slot_count=slot_count,
        feature_count=len(doc.features),
        total_points=total_points,
        orphan_points=orphan_points,
        mean_slot_load=round(mean_load, 2),
        median_slot_load=round(median_load, 2),
        min_slot_load=min_load,
        max_slot_load=max_load,
        p95_slot_load=round(p95_load, 2),
        std_slot_load=round(std_load, 2),
        imbalance_ratio=round(imbalance, 4),
        gini_coefficient=round(gini, 4),
        balance_threshold=balance_threshold,
        needs_rebalance=needs_rebalance,
        active_moves=active_moves,
        slots=slot_rows,
        features=feature_rows,
        size_histogram=histogram,
        split_recommendations=splits,
    )


def recommend_splits(
    doc: MapDoc,
    snapshot: LoadSnapshot | None = None,
    balance_threshold: float = BALANCE_RATIO,
) -> list[SplitRecommendation]:
    """Recommend features to split when a single feature exceeds slot load target."""
    slot_points, feature_totals, _per_slot, _orphans = _resolve_counts(doc, snapshot)
    slot_count = len(doc.slots)
    mean_load = (sum(slot_points.values()) / slot_count) if slot_count else 0.0
    return _recommend_splits(
        doc, feature_totals, mean_load, doc.busy_slots(), balance_threshold
    )


def _resolve_counts(
    doc: MapDoc, snapshot: LoadSnapshot | None
) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]], int]:
    per_slot: dict[str, dict[str, int]]
    if snapshot is None:
        slot_points = {slot: int(doc.slot_load.get(slot, 0)) for slot in doc.slots}
        feature_totals = {
            name: int(info.points_count) for name, info in doc.features.items()
        }
        per_slot = {
            name: _estimated_per_slot(info, feature_totals[name])
            for name, info in doc.features.items()
        }
        return slot_points, feature_totals, per_slot, 0

    slot_points = {slot: 0 for slot in doc.slots}
    feature_totals = {name: 0 for name in doc.features}
    per_slot = {name: {} for name in doc.features}
    orphan_points = 0
    for slot in doc.slots:
        facets = snapshot.counts.get(slot, {})
        slot_points[slot] = sum(facets.values())
        for raw_feature, count in facets.items():
            found = doc.find(str(raw_feature))
            if found is None:
                orphan_points += count
                continue
            canonical = found[0]
            feature_totals[canonical] += count
            per_slot[canonical][slot] = per_slot[canonical].get(slot, 0) + count
    return slot_points, feature_totals, per_slot, orphan_points


def _estimated_per_slot(info: FeatureInfo, total: int) -> dict[str, int]:
    if not info.shard_keys:
        return {}
    base, extra = divmod(total, len(info.shard_keys))
    return {
        slot: base + (1 if idx < extra else 0)
        for idx, slot in enumerate(info.shard_keys)
    }


def _feature_row(
    name: str,
    info: FeatureInfo,
    points: int,
    slot_counts: dict[str, int],
    total_points: int,
    now: datetime,
) -> FeatureStats:
    grace_iso = None
    if info.grace_until is not None:
        text = info.grace_until.astimezone(timezone.utc).isoformat()
        grace_iso = text[:-6] + "Z" if text.endswith("+00:00") else text
    return FeatureStats(
        name=name,
        points=points,
        share_pct=_share(points, total_points),
        shard_keys=tuple(info.shard_keys),
        width=len(info.shard_keys),
        state=info.state,
        write_to=None if info.write_to is None else tuple(info.write_to),
        aliases=tuple(info.aliases),
        copied=info.copied,
        grace_until=grace_iso,
        remaining_grace_seconds=remaining_grace_seconds(info, now),
        per_slot_points=slot_counts,
    )


def _recommend_splits(
    doc: MapDoc,
    feature_totals: dict[str, int],
    mean_load: float,
    busy: set[str],
    balance_threshold: float,
) -> list[SplitRecommendation]:
    if mean_load <= 0:
        return []
    limit = balance_threshold * mean_load
    results: list[SplitRecommendation] = []
    for name, info in doc.features.items():
        if info.state != FeatureState.stable or not info.shard_keys:
            continue
        points = feature_totals.get(name, 0)
        width = len(info.shard_keys)
        per_slot = points / width
        if per_slot <= limit:
            continue
        free_slots = [
            slot
            for slot in doc.slots
            if slot not in busy and slot not in info.shard_keys
        ]
        needed = math.ceil(points / mean_load)
        partitions = min(len(free_slots), needed)
        if partitions <= width:
            continue
        results.append(
            SplitRecommendation(
                feature=name,
                current_width=width,
                recommended_partitions=partitions,
                points=points,
                reason=(
                    f"{round(per_slot)} points/slot exceeds "
                    f"{balance_threshold}x mean ({round(limit)})"
                ),
            )
        )
    results.sort(key=lambda item: (-item.points, item.feature))
    return results


def _build_histogram(features: tuple[FeatureStats, ...]) -> tuple[HistogramBucket, ...]:
    buckets: list[HistogramBucket] = []
    for label, low, high in _HISTOGRAM_BOUNDS:
        count = 0
        total = 0
        for row in features:
            if row.points < low:
                continue
            if high is not None and row.points > high:
                continue
            count += 1
            total += row.points
        buckets.append(HistogramBucket(label, low, high, count, total))
    return tuple(buckets)


def _slot_status(
    points: int, busy: bool, mean_load: float, threshold: float
) -> str:
    if busy:
        return "busy"
    if points == 0:
        return "empty"
    if mean_load > 0 and points > threshold * mean_load:
        return "hot"
    return "balanced"


def _share(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((value / total) * 100.0, 2)


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _gini(values: list[int]) -> float:
    total = sum(values)
    n = len(values)
    if n <= 1 or total <= 0:
        return 0.0
    ordered = sorted(values)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n
