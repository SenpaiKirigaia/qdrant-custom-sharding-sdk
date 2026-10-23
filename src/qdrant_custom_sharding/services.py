import copy
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar

from qdrant_custom_sharding.errors import InvalidMove, VersionConflict
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc
from qdrant_custom_sharding.planning import plan_placement
from qdrant_custom_sharding.ports import MapStore, VectorStore
from qdrant_custom_sharding.routing import pick_slot

T = TypeVar("T")


def update_map(
    store: MapStore,
    change: Callable[[MapDoc], T],
    retries: int = 5,
) -> T:
    conflict: VersionConflict | None = None
    for _ in range(retries):
        doc = copy.deepcopy(store.load())
        result = change(doc)
        doc.version += 1
        try:
            store.save(doc)
        except VersionConflict as exc:
            conflict = exc
            continue
        return result
    if conflict is None:
        raise VersionConflict("map update failed")
    raise conflict


_FEATURE_TYPES = {"keyword", "integer"}


def check_feature_type(feature_type: str) -> None:
    if feature_type not in _FEATURE_TYPES:
        raise ValueError("feature_type must be keyword or integer")


def build_collection_map(
    collection: str,
    feature_field: str,
    feature_type: str,
    n_slots: int,
    feature_counts: dict[str, int] | None,
) -> MapDoc:
    slots = [f"s{index}" for index in range(n_slots)]
    features: dict[str, FeatureInfo] = {}
    slot_load = {slot: 0 for slot in slots}
    if feature_counts is not None:
        placement = plan_placement(feature_counts, slots)
        features = {
            name: FeatureInfo(
                shard_keys=list(keys),
                points_count=feature_counts[name],
            )
            for name, keys in placement.items()
        }
        slot_load = _spread_load(placement, feature_counts, slots)
    return MapDoc(
        collection=collection,
        feature_field=feature_field,
        slots=slots,
        features=features,
        slot_load=slot_load,
        feature_type=feature_type,
        version=1,
    )


@dataclass(frozen=True)
class CopyJob:

    collection: str
    field: str
    old_slots: list[str]
    targets: list[str]
    values: list[int | str]


def copy_job(doc: MapDoc, feature: int | str) -> CopyJob:
    found = doc.find(feature)
    if found is None:
        raise InvalidMove(f"unknown feature {feature}")
    name, info = found
    if not info.write_to:
        raise InvalidMove("copy requires write_to")
    return CopyJob(
        doc.collection,
        doc.feature_field,
        list(info.shard_keys),
        list(info.write_to),
        doc.payload_values(name),
    )


def group_by_slot(points, targets: list[str]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for point in points:
        grouped.setdefault(pick_slot(targets, point.id), []).append(point)
    return grouped


@dataclass(frozen=True)
class AlignAction:

    insert_ids: list
    delete_ids: list


def present_ids(records) -> set:
    return {point.id for point in records}


def align_actions(ids: list, first: set, second: set | None = None) -> AlignAction:
    if second is None:
        return AlignAction(
            [point_id for point_id in ids if point_id in first],
            [point_id for point_id in ids if point_id not in first],
        )
    insert_ids = []
    delete_ids = []
    for point_id in ids:
        was_there = point_id in first
        there_now = point_id in second
        if was_there == there_now:
            continue
        if there_now:
            insert_ids.append(point_id)
        else:
            delete_ids.append(point_id)
    return AlignAction(insert_ids, delete_ids)


def commit_plan(records, action: AlignAction) -> tuple[list, list]:
    chosen = set(action.insert_ids)
    points = [point for point in records if point.id in chosen]
    return points, list(action.delete_ids)


_ID_PAGE = 256


def align_ids(vectors: VectorStore, collection: str, old_slot: str, new_slot: str, ids: list) -> set:
    if not ids:
        return set()
    wanted = list(ids)
    first = list(vectors.read_ids(collection, old_slot, wanted))
    _apply_align(vectors, collection, new_slot, first, align_actions(wanted, present_ids(first)))
    second = list(vectors.read_ids(collection, old_slot, wanted))
    _apply_align(
        vectors,
        collection,
        new_slot,
        second,
        align_actions(wanted, present_ids(first), present_ids(second)),
    )
    return present_ids(second)


def _apply_align(vectors: VectorStore, collection: str, slot: str, records, action: AlignAction) -> None:
    points, delete = commit_plan(records, action)
    vectors.insert_missing(collection, slot, points)
    if delete:
        vectors.delete_ids(collection, slot, delete)


@dataclass(frozen=True)
class SlotPair:

    old_slot: str
    new_slot: str
    ids: list


def reconcile_pairs(
    old_ids: set, new_ids: set, old_slots: list[str], new_slots: list[str]
) -> list[SlotPair]:
    buckets: dict[tuple[str, str], list] = {}
    order: list[tuple[str, str]] = []
    for point_id in sorted(set(old_ids) ^ set(new_ids), key=str):
        key = (pick_slot(old_slots, point_id), pick_slot(new_slots, point_id))
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(point_id)
    return [SlotPair(key[0], key[1], buckets[key]) for key in order]


def reconcile_tally(only_old: set, only_new: set, ids: list, present: set) -> tuple[int, int]:
    copied = 0
    removed = 0
    for point_id in ids:
        if point_id in only_old and point_id in present:
            copied += 1
        elif point_id in only_new and point_id not in present:
            removed += 1
    return copied, removed


def feature_ids(vectors: VectorStore, collection: str, slots: list[str], field: str, values: list) -> set:
    found: set = set()
    for slot in slots:
        for page in vectors.scroll_ids(collection, slot, field, values, _ID_PAGE):
            found.update(page)
    return found


@dataclass(frozen=True)
class ReconcileCounts:

    copied: int
    removed: int


def reconcile_scope(doc: MapDoc, feature: int | str):
    found = doc.find(feature)
    if found is None:
        raise InvalidMove(f"unknown feature {feature}")
    name, info = found
    if not info.write_to:
        raise InvalidMove("reconcile requires write_to")
    return (
        doc.collection,
        doc.feature_field,
        doc.payload_values(name),
        list(info.shard_keys),
        list(info.write_to),
    )


def reconcile_feature(vectors: VectorStore, doc: MapDoc, feature: int | str) -> ReconcileCounts:
    collection, field, values, old_slots, new_slots = reconcile_scope(doc, feature)
    old_ids = feature_ids(vectors, collection, old_slots, field, values)
    new_ids = feature_ids(vectors, collection, new_slots, field, values)
    only_old = old_ids - new_ids
    only_new = new_ids - old_ids
    copied = 0
    removed = 0
    for pair in reconcile_pairs(old_ids, new_ids, old_slots, new_slots):
        present = align_ids(vectors, doc.collection, pair.old_slot, pair.new_slot, pair.ids)
        add_copied, add_removed = reconcile_tally(only_old, only_new, pair.ids, present)
        copied += add_copied
        removed += add_removed
    return ReconcileCounts(copied, removed)


@dataclass(frozen=True)
class CutoverResult:

    old_slots: list[str]
    copied: int
    removed: int


@dataclass(frozen=True)
class FinishJob:

    collection: str
    field: str
    old_slots: list[str]
    values: list[int | str]


def finish_job(doc: MapDoc, feature: int | str, now: datetime) -> FinishJob:
    preview = copy.deepcopy(doc)
    preview.finish_move(feature, now)
    found = doc.find(feature)
    if found is None:
        raise InvalidMove(f"unknown feature {feature}")
    name, info = found
    if not info.write_to:
        raise InvalidMove("finish_move requires write_to")
    targets = set(info.write_to)
    old_slots = [slot for slot in info.shard_keys if slot not in targets]
    return FinishJob(
        doc.collection,
        doc.feature_field,
        old_slots,
        doc.payload_values(name),
    )


def apply_stats(doc: MapDoc, counts: dict[str, dict[str, int]]) -> MapDoc:
    doc.set_stats(counts)
    return doc


def create_sharded_collection(
    store: MapStore,
    vectors: VectorStore,
    collection: str,
    feature_field: str,
    feature_type: str,
    n_slots: int = 48,
    feature_counts: dict[str, int] | None = None,
    **collection_kwargs: object,
) -> MapDoc:
    check_feature_type(feature_type)
    vectors.create_collection(collection, **collection_kwargs)
    doc = build_collection_map(
        collection, feature_field, feature_type, n_slots, feature_counts
    )
    for slot in doc.slots:
        vectors.create_slot(collection, slot)
    vectors.create_feature_index(collection, feature_field, feature_type)
    store.save(doc)
    return doc


def add_features(store: MapStore, values: list[int | str]) -> MapDoc:

    def change(doc: MapDoc) -> MapDoc:
        for value in values:
            doc.add_feature(value)
        return doc

    return update_map(store, change)


def add_feature(store: MapStore, value: int | str) -> FeatureInfo:

    def change(doc: MapDoc) -> FeatureInfo:
        return doc.add_feature(value)

    return update_map(store, change)


def start_move(
    store: MapStore,
    feature: int | str,
    targets: list[str],
    now: datetime,
    grace_seconds: int,
) -> None:
    grace_until = now + timedelta(seconds=grace_seconds)

    def change(doc: MapDoc) -> None:
        doc.start_move(feature, targets, grace_until)

    update_map(store, change)


def copy_feature(
    store: MapStore,
    vectors: VectorStore,
    feature: int | str,
    now: datetime,
    batch: int = 256,
) -> int:
    doc = store.load()
    doc.copy_allowed(feature, now)
    job = copy_job(doc, feature)
    copied = 0
    for slot in job.old_slots:
        for chunk in vectors.scroll(job.collection, slot, job.field, job.values, batch):
            copied += _copy_chunk(vectors, job, slot, chunk)

    def change(current: MapDoc) -> None:
        current.mark_copied(feature, now)

    update_map(store, change)
    return copied


def _copy_chunk(vectors: VectorStore, job: CopyJob, slot: str, chunk) -> int:
    count = 0
    for target, points in group_by_slot(chunk, job.targets).items():
        align_ids(vectors, job.collection, slot, target, [point.id for point in points])
        count += len(points)
    return count


def open_dual(
    store: MapStore,
    feature: int | str,
    now: datetime,
    grace_seconds: int,
) -> None:
    grace_until = now + timedelta(seconds=grace_seconds)

    def change(doc: MapDoc) -> None:
        doc.open_dual(feature, now, grace_until)

    update_map(store, change)


def cutover(
    store: MapStore,
    vectors: VectorStore,
    feature: int | str,
    now: datetime,
    grace_seconds: int,
) -> CutoverResult:
    doc = store.load()
    copy.deepcopy(doc).cutover(feature, now, now)
    counts = reconcile_feature(vectors, doc, feature)
    grace_until = now + timedelta(seconds=grace_seconds)

    def change(current: MapDoc) -> list[str]:
        return current.cutover(feature, now, grace_until)

    old_slots = update_map(store, change)
    return CutoverResult(old_slots, counts.copied, counts.removed)


def finish_move(
    store: MapStore,
    vectors: VectorStore,
    feature: int | str,
    now: datetime,
) -> None:
    job = finish_job(store.load(), feature, now)
    for slot in job.old_slots:
        vectors.delete_feature(job.collection, slot, job.field, job.values)

    def change(doc: MapDoc) -> None:
        doc.finish_move(feature, now)

    update_map(store, change)


def migrate(
    store: MapStore,
    vectors: VectorStore,
    feature: int | str,
    to_slots: list[str],
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[int], None],
    grace_seconds: int,
) -> None:
    start_move(store, feature, to_slots, now_fn(), grace_seconds)
    sleep_fn(grace_seconds)
    copy_feature(store, vectors, feature, now_fn())
    open_dual(store, feature, now_fn(), grace_seconds)
    sleep_fn(grace_seconds)
    cutover(store, vectors, feature, now_fn(), grace_seconds)
    sleep_fn(grace_seconds)
    finish_move(store, vectors, feature, now_fn())


def split_targets(doc: MapDoc, feature: int | str, partitions: int) -> list[str]:
    found = doc.find(feature)
    if found is None:
        raise InvalidMove(f"unknown feature {feature}")
    exclude = doc.busy_slots() | set(found[1].shard_keys)
    return doc.least_loaded(partitions, exclude)


def alias_already_renamed(doc: MapDoc, old: int | str, new: int | str) -> bool:
    found = doc.find(old)
    if found is None:
        raise InvalidMove(f"unknown feature {old}")
    canonical, info = found
    if info.state != FeatureState.stable:
        raise InvalidMove("alias_feature requires state stable")
    old_name = str(old)
    new_name = str(new)
    if canonical == old_name:
        if doc.find(new_name) is not None:
            raise InvalidMove(f"{new_name} is already in use")
        return False
    if canonical != new_name:
        raise InvalidMove(f"{new_name} is already in use")
    return True


def alias_feature(
    store: MapStore,
    vectors: VectorStore,
    old: int | str,
    new: int | str,
    sleep_fn: Callable[[int], None],
    grace_seconds: int,
) -> int:
    if not alias_already_renamed(store.load(), old, new):

        def change(doc: MapDoc) -> None:
            doc.alias_feature(old, new)

        update_map(store, change)
    sleep_fn(grace_seconds)
    doc = store.load()
    found = doc.find(new)
    if found is None or found[1].state != FeatureState.stable:
        raise InvalidMove("alias_feature requires state stable")
    return _rewrite_aliases(vectors, doc, found[0])


def drop_alias(store: MapStore, vectors: VectorStore, old: int | str) -> int:
    doc = store.load()
    name = _alias_name(doc, old)
    count = _rewrite_aliases(vectors, doc, name)

    def change(current: MapDoc) -> None:
        current.drop_alias(old)

    update_map(store, change)
    return count


def _alias_name(doc: MapDoc, old: int | str) -> str:
    old_name = str(old)
    for name, info in doc.features.items():
        if old_name not in info.aliases:
            continue
        if info.state != FeatureState.stable:
            raise InvalidMove("drop_alias requires state stable")
        return name
    raise InvalidMove(f"unknown alias {old_name}")


def _rewrite_aliases(vectors: VectorStore, doc: MapDoc, name: str) -> int:
    info = doc.features[name]
    typed = doc.payload_values(name)
    return vectors.rewrite_feature(
        doc.collection,
        list(info.shard_keys),
        doc.feature_field,
        list(typed[1:]),
        typed[0],
    )


def refresh_stats(store: MapStore, vectors: VectorStore) -> MapDoc:

    def change(doc: MapDoc) -> MapDoc:
        counts = {
            slot: vectors.feature_counts(doc.collection, slot, doc.feature_field)
            for slot in doc.slots
        }
        return apply_stats(doc, counts)

    return update_map(store, change)


def _spread_load(
    placement: dict[str, list[str]],
    counts: dict[str, int],
    slots: list[str],
) -> dict[str, int]:
    load = {slot: 0 for slot in slots}
    for feature, chosen in placement.items():
        base, extra = divmod(counts[feature], len(chosen))
        for index, slot in enumerate(chosen):
            load[slot] += base + (1 if index < extra else 0)
    return load
