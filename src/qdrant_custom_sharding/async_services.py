import asyncio
import copy
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from qdrant_custom_sharding.errors import InvalidMove, VersionConflict
from qdrant_custom_sharding.load import LoadSnapshot, snapshot_from_facets
from qdrant_custom_sharding.models import FeatureState, MapDoc, remaining_grace_seconds
from qdrant_custom_sharding.planning import RebalancePlan, plan_rebalance as build_rebalance_plan
from qdrant_custom_sharding.ports import MapStore
from qdrant_custom_sharding.qdrant_store import AsyncQdrantVectorStore
from qdrant_custom_sharding.services import (
    _ID_PAGE,
    _alias_name,
    _rebalance_action,
    CutoverResult,
    align_actions,
    apply_stats,
    build_collection_map,
    check_feature_type,
    check_grace_seconds,
    check_n_slots,
    check_slot_name,
    commit_plan,
    copy_job,
    finish_job,
    group_by_slot,
    open_dual,
    present_ids,
    reconcile_pairs,
    reconcile_scope,
    reconcile_tally,
    alias_already_renamed,
    start_move,
    update_map,
)


async def create_sharded_collection(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    collection: str,
    feature_field: str,
    feature_type: str,
    n_slots: int = 48,
    feature_counts: dict[str, int] | None = None,
    **collection_kwargs: object,
) -> MapDoc:
    check_feature_type(feature_type)
    check_n_slots(n_slots)
    await vectors.create_collection(collection, **collection_kwargs)
    doc = build_collection_map(
        collection, feature_field, feature_type, n_slots, feature_counts
    )
    for slot in doc.slots:
        await vectors.create_slot(collection, slot)
    await vectors.create_feature_index(collection, feature_field, feature_type)
    await asyncio.to_thread(store.save, doc)
    return doc


async def add_slot(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    slot: str,
) -> MapDoc:
    check_slot_name(slot)
    doc = await asyncio.to_thread(store.load)
    if slot in doc.slots:
        return doc
    await vectors.create_slot(doc.collection, slot)

    def change(current: MapDoc) -> MapDoc:
        current.add_slot(slot)
        return current

    return await asyncio.to_thread(update_map, store, change)


async def copy_feature(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    feature: int | str,
    now: datetime,
    batch: int = 256,
) -> int:
    doc = await asyncio.to_thread(store.load)
    doc.copy_allowed(feature, now)
    job = copy_job(doc, feature)
    copied = 0
    for slot in job.old_slots:
        async for chunk in vectors.scroll(
            job.collection, slot, job.field, job.values, batch
        ):
            for target, points in group_by_slot(chunk, job.targets).items():
                await align_ids(
                    vectors, job.collection, slot, target, [point.id for point in points]
                )
                copied += len(points)

    def change(current: MapDoc) -> None:
        current.mark_copied(feature, now)

    await asyncio.to_thread(update_map, store, change)
    return copied


async def align_ids(vectors, collection: str, old_slot: str, new_slot: str, ids: list) -> set:
    if not ids:
        return set()
    wanted = list(ids)
    first = list(await vectors.read_ids(collection, old_slot, wanted))
    await _apply_align(
        vectors, collection, new_slot, first, align_actions(wanted, present_ids(first))
    )
    second = list(await vectors.read_ids(collection, old_slot, wanted))
    await _apply_align(
        vectors,
        collection,
        new_slot,
        second,
        align_actions(wanted, present_ids(first), present_ids(second)),
    )
    return present_ids(second)


async def _apply_align(vectors, collection: str, slot: str, records, action) -> None:
    points, delete = commit_plan(records, action)
    await vectors.insert_missing(collection, slot, points)
    if delete:
        await vectors.delete_ids(collection, slot, delete)


async def _feature_ids(vectors, collection: str, slots: list[str], field: str, values: list) -> set:
    found: set = set()
    for slot in slots:
        async for page in vectors.scroll_ids(collection, slot, field, values, _ID_PAGE):
            found.update(page)
    return found


async def reconcile_feature(vectors, doc: MapDoc, feature: int | str):
    collection, field, values, old_slots, new_slots = reconcile_scope(doc, feature)
    old_ids = await _feature_ids(vectors, collection, old_slots, field, values)
    new_ids = await _feature_ids(vectors, collection, new_slots, field, values)
    only_old = old_ids - new_ids
    only_new = new_ids - old_ids
    copied = 0
    removed = 0
    for pair in reconcile_pairs(old_ids, new_ids, old_slots, new_slots):
        present = await align_ids(vectors, collection, pair.old_slot, pair.new_slot, pair.ids)
        add_copied, add_removed = reconcile_tally(only_old, only_new, pair.ids, present)
        copied += add_copied
        removed += add_removed
    return copied, removed


async def cutover(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    feature: int | str,
    now: datetime,
    grace_seconds: int,
) -> CutoverResult:
    check_grace_seconds(grace_seconds)
    doc = await asyncio.to_thread(store.load)
    copy.deepcopy(doc).cutover(feature, now, now)
    copied, removed = await reconcile_feature(vectors, doc, feature)
    grace_until = now + timedelta(seconds=grace_seconds)

    def change(current: MapDoc) -> list[str]:
        return current.cutover(feature, now, grace_until)

    old_slots = await asyncio.to_thread(update_map, store, change)
    return CutoverResult(old_slots, copied, removed)


async def finish_move(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    feature: int | str,
    now: datetime,
) -> None:
    job = finish_job(await asyncio.to_thread(store.load), feature, now)
    for slot in job.old_slots:
        await vectors.delete_feature(job.collection, slot, job.field, job.values)

    def change(doc: MapDoc) -> None:
        doc.finish_move(feature, now)

    await asyncio.to_thread(update_map, store, change)


async def resume_move(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    feature: int | str,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[int], Awaitable[object]],
    grace_seconds: int,
) -> None:
    check_grace_seconds(grace_seconds)
    while True:
        found = (await asyncio.to_thread(store.load)).find(feature)
        if found is None:
            raise InvalidMove(f"unknown feature {feature}")
        info = found[1]
        if info.state == FeatureState.stable:
            return
        if info.grace_until is None:
            raise InvalidMove("resume_move requires grace_until")
        if info.state == FeatureState.copying and not info.copied:
            await sleep_fn(remaining_grace_seconds(info, now_fn()))
            await copy_feature(store, vectors, feature, now_fn())
        elif info.state == FeatureState.copying:
            await asyncio.to_thread(open_dual, store, feature, now_fn(), grace_seconds)
        elif info.state == FeatureState.dual:
            await sleep_fn(remaining_grace_seconds(info, now_fn()))
            await cutover(store, vectors, feature, now_fn(), grace_seconds)
        elif info.state == FeatureState.cutover:
            await sleep_fn(remaining_grace_seconds(info, now_fn()))
            await finish_move(store, vectors, feature, now_fn())
            return


async def migrate(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    feature: int | str,
    to_slots: list[str],
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[int], Awaitable[object]],
    grace_seconds: int,
) -> None:
    await asyncio.to_thread(start_move, store, feature, to_slots, now_fn(), grace_seconds)
    await resume_move(store, vectors, feature, now_fn, sleep_fn, grace_seconds)


async def _facet_snapshot(
    vectors: AsyncQdrantVectorStore,
    doc: MapDoc,
) -> LoadSnapshot:
    counts: dict[str, dict[str, int]] = {}
    for slot in doc.slots:
        counts[slot] = await vectors.feature_counts(
            doc.collection, slot, doc.feature_field
        )
    return snapshot_from_facets(doc.collection, doc.slots, counts)


async def read_load(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
) -> LoadSnapshot:
    doc = await asyncio.to_thread(store.load)
    return await _facet_snapshot(vectors, doc)


async def plan_rebalance(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
) -> RebalancePlan:
    doc = await asyncio.to_thread(store.load)
    snapshot = await _facet_snapshot(vectors, doc)
    return build_rebalance_plan(snapshot, doc)


async def apply_rebalance(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    plan: RebalancePlan,
    now_fn: Callable[[], datetime],
    sleep_fn: Callable[[int], Awaitable[object]],
    grace_seconds: int,
) -> None:
    check_grace_seconds(grace_seconds)
    doc = await asyncio.to_thread(store.load)
    if plan.collection != doc.collection:
        raise InvalidMove("plan collection does not match the map")
    if not plan.moves:
        return
    for order in plan.moves:
        found = (await asyncio.to_thread(store.load)).find(order.feature)
        if found is None:
            raise InvalidMove(f"unknown feature {order.feature}")
        action = _rebalance_action(order.feature, found[1], order.to_slots)
        if action == "migrate":
            await migrate(
                store,
                vectors,
                order.feature,
                list(order.to_slots),
                now_fn,
                sleep_fn,
                grace_seconds,
            )
        elif action == "resume":
            await resume_move(
                store, vectors, order.feature, now_fn, sleep_fn, grace_seconds
            )


async def refresh_stats(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    retries: int = 5,
) -> MapDoc:
    conflict: VersionConflict | None = None
    for _ in range(retries):
        snapshot = await asyncio.to_thread(store.load)
        loaded = await _facet_snapshot(vectors, snapshot)
        seen = snapshot.version

        def change(doc: MapDoc, loaded=loaded, seen=seen) -> MapDoc:
            if doc.version != seen:
                raise VersionConflict("map changed while counting")
            return apply_stats(doc, loaded.as_facet_counts())

        try:
            return await asyncio.to_thread(update_map, store, change, 1)
        except VersionConflict as exc:
            conflict = exc
    if conflict is None:
        raise VersionConflict("map update failed")
    raise conflict


async def alias_feature(
    store: MapStore,
    vectors: AsyncQdrantVectorStore,
    old: int | str,
    new: int | str,
    sleep_fn: Callable[[int], Awaitable[object]],
    grace_seconds: int,
) -> int:
    check_grace_seconds(grace_seconds)
    doc = await asyncio.to_thread(store.load)
    if not alias_already_renamed(doc, old, new):

        def change(current: MapDoc) -> None:
            current.alias_feature(old, new)

        await asyncio.to_thread(update_map, store, change)
    await sleep_fn(grace_seconds)
    doc = await asyncio.to_thread(store.load)
    found = doc.find(new)
    if found is None or found[1].state != FeatureState.stable:
        raise InvalidMove("alias_feature requires state stable")
    return await _rewrite_aliases(vectors, doc, found[0])


async def drop_alias(
    store: MapStore, vectors: AsyncQdrantVectorStore, old: int | str
) -> int:
    doc = await asyncio.to_thread(store.load)
    name = _alias_name(doc, old)
    count = await _rewrite_aliases(vectors, doc, name)

    def change(current: MapDoc) -> None:
        current.drop_alias(old)

    await asyncio.to_thread(update_map, store, change)
    return count


async def _rewrite_aliases(vectors: AsyncQdrantVectorStore, doc: MapDoc, name: str) -> int:
    info = doc.features[name]
    typed = doc.payload_values(name)
    return await vectors.rewrite_feature(
        doc.collection,
        list(info.shard_keys),
        doc.feature_field,
        list(typed[1:]),
        typed[0],
    )
