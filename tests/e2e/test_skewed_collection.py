import random
import threading
import time
import uuid
import warnings
import zlib

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from qdrant_custom_sharding import FanOutWarning, FeatureMovedWarning, FeatureState
from tests.e2e.conftest import open_client

pytestmark = pytest.mark.e2e

_SEED = 20261013
_DIM = 16
_VECTOR = [0.1] * _DIM
_INT_FEATURES = 500
_INT_POINTS = 30_000
_INT_SLOTS = 48
_STR_FEATURES = 50
_STR_POINTS = 2_500
_STR_SLOTS = 12


def _zipf_counts(n_features: int, n_points: int, exponent: float, seed: int) -> list[int]:
    weights = [1.0 / (rank ** exponent) for rank in range(1, n_features + 1)]
    remaining = n_points - n_features
    scale = remaining / sum(weights)
    raw = [weight * scale for weight in weights]
    extra = [int(value) for value in raw]
    left = remaining - sum(extra)
    rng = random.Random(seed)
    tie = [rng.random() for _ in range(n_features)]
    order = sorted(
        range(n_features),
        key=lambda index: (raw[index] - extra[index], tie[index]),
        reverse=True,
    )
    for index in range(left):
        extra[order[index]] += 1
    return [1 + count for count in extra]


def _named_counts(n_features, n_points, exponent, seed, prefix):
    raw = _zipf_counts(n_features, n_points, exponent, seed)
    if prefix is None:
        return {str(index + 1): raw[index] for index in range(n_features)}
    return {f"{prefix}{index:02d}": raw[index] for index in range(n_features)}


def _points(field, counts, as_int):
    points = []
    ranges = {}
    point_id = 1
    order = sorted(counts, key=lambda name: (-counts[name], name))
    for feature in order:
        count = counts[feature]
        ranges[feature] = range(point_id, point_id + count)
        value = int(feature) if as_int else feature
        for _ in range(count):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(_VECTOR),
                    payload={field: value},
                )
            )
            point_id += 1
    return points, ranges


def _load(client, name, points):
    tail = min(500, max(1, len(points) // 5))
    if tail >= len(points):
        tail = len(points) // 2
    client.upload_points(name, points[:-tail], batch_size=256, wait=True)
    client.upsert(name, points[-tail:], wait=True)
    return len(points) - tail, tail


def _create(client, name, field, feature_type, n_slots, counts):
    return client.sharding.create_collection(
        name,
        field,
        feature_type=feature_type,
        n_slots=n_slots,
        feature_counts=counts,
        vectors_config=VectorParams(size=_DIM, distance=Distance.COSINE),
    )


def _filter(field, value):
    return Filter(must=FieldCondition(key=field, match=MatchValue(value=value)))


def _count(client, name, field, value, slot=None):
    kwargs = {"count_filter": _filter(field, value), "exact": True}
    if slot is not None:
        kwargs["shard_key_selector"] = slot
    return client.count(name, **kwargs).count


def _sample(counts, seed, top_n, random_n):
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    rest = ranked[top_n:]
    rng = random.Random(seed)
    rng.shuffle(rest)
    return ranked[:top_n] + rest[:random_n]


def _value(feature, as_int):
    return int(feature) if as_int else feature


def _check_features(client, name, field, counts, features, slots, as_int):
    for feature in features:
        value = _value(feature, as_int)
        expected = counts[feature]
        counted = _count(client, name, field, value)
        assert counted == expected, f"{field}={value}: count {counted} != {expected}"
        found = client.query_points(
            name, _VECTOR, query_filter=_filter(field, value), limit=24
        )
        got = {point.payload[field] for point in found.points}
        assert got == {value}, f"{field}={value}: query returned {got}"
        total = sum(_count(client, name, field, value, slot) for slot in slots)
        assert total == expected, f"{field}={value}: slot sum {total} != {expected}"


def _check_fanout(client, name, n_points):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total = client.count(name, exact=True).count
    assert total == n_points, f"fan-out count {total} != {n_points}"
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert fan, "read without a feature filter did not warn FanOutWarning"
    return total


def _hash_counts(point_ids, slots):
    expected = {slot: 0 for slot in slots}
    width = len(slots)
    for point_id in point_ids:
        index = zlib.crc32(str(point_id).encode()) % width
        expected[slots[index]] += 1
    return expected


def _row(step, started, result):
    seconds = time.perf_counter() - started
    print(f"{step:<16} {seconds:7.2f}  {result}", flush=True)
    return time.perf_counter()


def _header(title):
    print(f"\n{title}", flush=True)
    print(f"{'step':<16} {'sec':>7}  result", flush=True)


def _write_during_migrate(admin, writer, name, value, target):
    stop = threading.Event()
    written = []
    errors = []

    def run():
        point_id = 90_000_000
        try:
            while not stop.is_set() and len(written) < 30:
                writer.upsert(
                    name,
                    [
                        PointStruct(
                            id=point_id,
                            vector=list(_VECTOR),
                            payload={"subject_id": value},
                        )
                    ],
                    wait=True,
                )
                written.append(point_id)
                point_id += 1
                if stop.wait(0.2):
                    return
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, name="e2e-writer", daemon=True)
    thread.start()
    try:
        admin.sharding.migrate(name, value, target, grace_seconds=3)
    finally:
        stop.set()
        thread.join(timeout=120)
    if thread.is_alive():
        raise RuntimeError("writer thread did not stop")
    if errors:
        raise errors[0]
    return written


def test_integer_feature_on_the_local_stack(cluster):
    client, map_uri, created = cluster
    name = "e2e" + uuid.uuid4().hex[:12]
    created.append(name)
    counts = _named_counts(_INT_FEATURES, _INT_POINTS, 1.05, _SEED, None)
    assert sum(counts.values()) == _INT_POINTS
    assert len(counts) == _INT_FEATURES
    top_share = max(counts.values()) / _INT_POINTS
    assert 0.15 <= top_share <= 0.20, f"top feature share {top_share:.3f}"
    points, ranges = _points("subject_id", counts, as_int=True)
    print(f"map {map_uri}", flush=True)
    _header(f"integer collection {name}")
    started = time.perf_counter()

    _create(client, name, "subject_id", "integer", _INT_SLOTS, counts)
    uploaded, upserted = _load(client, name, points)
    started = _row(
        "load",
        started,
        f"upload_points {uploaded} + upsert {upserted}, features {len(counts)}",
    )

    doc = client.sharding.refresh_stats(name)
    loads = [doc.slot_load[slot] for slot in doc.slots]
    total_load = sum(loads)
    mean = total_load / len(loads)
    top = max(counts, key=lambda feature: (counts[feature], feature))
    top_slots = len(doc.features[top].shard_keys)
    balance = (
        f"max {max(loads)} min {min(loads)} mean {mean:.1f} "
        f"ratio {max(loads) / mean:.3f} top {top} slots {top_slots} "
        f"share {counts[top] / _INT_POINTS:.3f}"
    )
    started = _row("balance", started, balance)
    assert total_load == _INT_POINTS, balance
    assert doc.features[top].points_count == counts[top], balance
    assert top_slots > 1, balance
    assert max(loads) <= 1.5 * mean, balance

    features = _sample(counts, _SEED, 5, 20)
    print(f"checking {len(features)} features on {len(doc.slots)} slots", flush=True)
    _check_features(
        client, name, "subject_id", counts, features, doc.slots, as_int=True
    )
    started = _row("sample", started, f"{len(features)} features, counts and slot sums match")

    fan = _check_fanout(client, name, _INT_POINTS)
    started = _row("fanout", started, f"count {fan}, FanOutWarning")

    single = [
        feature
        for feature, info in doc.features.items()
        if len(info.shard_keys) == 1
    ]
    single.sort(key=lambda feature: (counts[feature], feature))
    migrated_name = single[len(single) // 2]
    value = int(migrated_name)
    old_slot = doc.features[migrated_name].shard_keys[0]
    target = next(slot for slot in doc.slots if slot != old_slot)
    before = _count(client, name, "subject_id", value)
    assert before == counts[migrated_name]
    writer = open_client(map_uri, map_ttl_seconds=1)
    try:
        written = _write_during_migrate(client, writer, name, value, target)
    finally:
        writer.close()
    expected = before + len(written)
    moved = client.sharding.get_map(name)
    info = moved.features[migrated_name]
    slot_counts = {
        slot: _count(client, name, "subject_id", value, slot) for slot in moved.slots
    }
    migrate_text = (
        f"feature {value} {old_slot} -> {target}, "
        f"before {before} written {len(written)} after {sum(slot_counts.values())}"
    )
    started = _row("migrate", started, migrate_text)
    assert len(written) >= 1, migrate_text
    assert info.state == FeatureState.stable, migrate_text
    assert info.shard_keys == [target], migrate_text
    assert info.write_to is None
    assert slot_counts[target] == expected, migrate_text
    assert slot_counts[old_slot] == 0, migrate_text
    assert sum(slot_counts.values()) == expected, migrate_text

    stay = [
        feature
        for feature, row in moved.features.items()
        if feature != migrated_name and len(row.shard_keys) == 1
    ]
    split_name = max(stay, key=lambda feature: (counts[feature], feature))
    split_value = int(split_name)
    split_doc = client.sharding.split(name, split_value, 3, grace_seconds=0)
    split_info = split_doc.features[split_name]
    expected_slots = _hash_counts(ranges[split_name], split_info.shard_keys)
    actual = {
        slot: _count(client, name, "subject_id", split_value, slot)
        for slot in split_doc.slots
    }
    spread = {slot: actual[slot] for slot in split_info.shard_keys}
    split_text = (
        f"feature {split_value} -> {split_info.shard_keys}, "
        f"counts {spread}, sum {sum(actual.values())}"
    )
    started = _row("split", started, split_text)
    assert split_info.state == FeatureState.stable
    assert len(split_info.shard_keys) == 3, split_text
    assert split_info.write_to is None
    for slot in split_doc.slots:
        want = expected_slots.get(slot, 0)
        assert actual[slot] == want, f"{slot}: {actual[slot]} != {want}"
    assert sum(actual.values()) == counts[split_name]
    assert all(count > 0 for count in spread.values()), split_text

    alias_name = min(
        (
            feature
            for feature, row in split_doc.features.items()
            if feature not in {migrated_name, split_name}
            and feature in counts
            and len(row.shard_keys) == 1
        ),
        key=lambda feature: (counts[feature], feature),
    )
    old_id = int(alias_name)
    new_id = old_id + 1_000_000
    old_slots = list(split_doc.features[alias_name].shard_keys)
    changed = client.sharding.alias_feature(name, old_id, new_id, grace_seconds=0)
    aliased = client.sharding.get_map(name)
    assert aliased.features[str(new_id)].shard_keys == old_slots
    assert str(old_id) in aliased.features[str(new_id)].aliases
    assert changed == counts[alias_name]
    wrong = next(slot for slot in aliased.slots if slot not in old_slots)
    client.upsert(
        name,
        [PointStruct(id=80_000_001, vector=list(_VECTOR), payload={"subject_id": old_id})],
        shard_key_selector=wrong,
        wait=True,
    )
    old_visible = _count(client, name, "subject_id", old_id)
    assert old_visible == 0
    assert _count(client, name, "subject_id", old_id, wrong) == 1
    client.upsert(
        name,
        [PointStruct(id=80_000_003, vector=list(_VECTOR), payload={"subject_id": new_id})],
        wait=True,
    )
    client.upsert(
        name,
        [PointStruct(id=80_000_002, vector=list(_VECTOR), payload={"subject_id": new_id})],
        shard_key_selector=wrong,
        wait=True,
    )
    new_visible = _count(client, name, "subject_id", new_id)
    sample = client.retrieve(name, ids=[ranges[alias_name][0]])
    alias_text = (
        f"alias {old_id} -> {new_id}, same slots {old_slots}, "
        f"rewrote {changed}, "
        f"old-id filter {old_visible} (hides stray), "
        f"new-id filter {new_visible}"
    )
    started = _row("alias", started, alias_text)
    assert new_visible == counts[alias_name] + 1, alias_text
    assert sample[0].payload["subject_id"] == new_id
    assert _count(client, name, "subject_id", new_id, wrong) == 1

    fresh = client.sharding.get_map(name)
    pairs = [
        (feature, info)
        for feature, info in fresh.features.items()
        if feature in counts and feature not in {migrated_name, split_name, alias_name}
    ]
    pairs.sort(key=lambda item: (counts[item[0]], item[0]))
    source_name, source_info = pairs[0]
    dest_name, dest_info = next(
        item
        for item in pairs[1:]
        if set(item[1].shard_keys).isdisjoint(source_info.shard_keys)
    )
    source_value = int(source_name)
    dest_value = int(dest_name)
    hit = client.query_points(
        name, _VECTOR, query_filter=_filter("subject_id", source_value), limit=1
    )
    point_id = hit.points[0].id
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.upsert(
            name,
            [
                PointStruct(
                    id=point_id,
                    vector=list(_VECTOR),
                    payload={"subject_id": dest_value},
                )
            ],
            wait=True,
        )
    id_filter = Filter(must=[HasIdCondition(has_id=[point_id])])
    found_slots = []
    for slot in fresh.slots:
        n = client.count(
            name,
            count_filter=id_filter,
            shard_key_selector=slot,
            exact=True,
        ).count
        if n:
            found_slots.append((slot, n))
    move_text = (
        f"point {point_id} {source_value} -> {dest_value}, slots {found_slots}"
    )
    _row("move", started, move_text)
    assert any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    assert found_slots, move_text
    assert sum(count for _slot, count in found_slots) == 1, move_text
    home = found_slots[0][0]
    assert home in dest_info.shard_keys, move_text
    assert home not in source_info.shard_keys, move_text


def test_keyword_feature_on_the_local_stack(cluster):
    client, map_uri, created = cluster
    name = "e2e" + uuid.uuid4().hex[:12]
    created.append(name)
    counts = _named_counts(_STR_FEATURES, _STR_POINTS, 1.0, _SEED, "c")
    assert sum(counts.values()) == _STR_POINTS
    assert len(counts) == _STR_FEATURES
    points, _ranges = _points("category", counts, as_int=False)
    print(f"map {map_uri}", flush=True)
    _header(f"keyword collection {name}")
    started = time.perf_counter()

    doc = _create(client, name, "category", "keyword", _STR_SLOTS, counts)
    uploaded, upserted = _load(client, name, points)
    started = _row(
        "load",
        started,
        f"upload_points {uploaded} + upsert {upserted}, features {len(counts)}",
    )

    features = _sample(counts, _SEED + 1, 3, 5)
    print(f"checking {len(features)} features on {len(doc.slots)} slots", flush=True)
    _check_features(
        client, name, "category", counts, features, doc.slots, as_int=False
    )
    started = _row("sample", started, f"{len(features)} features, counts and slot sums match")

    fan = _check_fanout(client, name, _STR_POINTS)
    _row("fanout", started, f"count {fan}, FanOutWarning")
