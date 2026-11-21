from __future__ import annotations

import functools
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import ShardingMethod, UpdateResult, UpdateStatus

from qdrant_custom_sharding import services
from qdrant_custom_sharding.calls import (
    CALL_FANOUT,
    MAP_UNAVAILABLE,
    MAP_UNAVAILABLE_READ,
    PARALLEL_UPLOAD,
    READ_FANOUT,
    SELECTOR_FANOUT,
    STALE_USED,
    batch_fields,
    bind_call,
    copy_on_write,
    delete_ids,
    feature_retrieve,
    features_need_reload,
    full_retrieve,
    id_scroll_page,
    id_update_plan,
    is_partial_update,
    ids_by_slot,
    moved_upsert,
    operation_call,
    payload_moves_feature,
    points_for_records,
    points_with_payload,
    record_source,
    reject_clear,
    reject_feature_delete,
    request_on_slots,
    require_sdk_sharding,
    scroll_page,
    selector_ids,
    selector_with_ids,
    slot_upsert,
    stored_payloads,
    supported_kwargs,
    upload_upsert,
    upsert_wait,
    write_uses_all_slots,
)
from qdrant_custom_sharding.errors import MapNotFound, WriteDenied
from qdrant_custom_sharding.load import LoadSnapshot
from qdrant_custom_sharding.planning import RebalancePlan
from qdrant_custom_sharding.map_cache import MapCache, Reason
from qdrant_custom_sharding.map_stores import folder_from_uri, list_map_collections
from qdrant_custom_sharding.metrics import InMemoryMetrics, MetricsRecorder
from qdrant_custom_sharding.models import MapDoc
from qdrant_custom_sharding.ports import MapFolder
from qdrant_custom_sharding.qdrant_store import QdrantVectorStore
from qdrant_custom_sharding.stats import CollectionStats, collection_stats
from qdrant_custom_sharding.router import (
    chunks,
    filter_copy_targets,
    iter_upload_points,
    moved_points,
    mutation_slots,
    needs_reload,
    plan_writes,
    point_structs,
    read_selector,
    selector_filter,
)
from qdrant_custom_sharding.sharding_warnings import (
    FanOutWarning,
    FeatureMovedWarning,
    ShardingWarning,
    StaleMapWarning,
    warn,
)

_MAP_STACKLEVEL = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ShardedQdrantClient(QdrantClient):
    """QdrantClient that routes a collection when its map file exists."""

    def __init__(
        self,
        *args: object,
        map_uri: str | None = None,
        map_ttl_seconds: float = 30,
        sharding_warnings: bool = True,
        detect_feature_changes: bool = True,
        map_folder: MapFolder | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[int], None] | None = None,
        metrics: MetricsRecorder | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._router = _CallRouter(
            self,
            map_folder if map_folder is not None else folder_from_uri(map_uri),
            map_ttl_seconds,
            sharding_warnings,
            detect_feature_changes,
            now_fn or _now,
            sleep_fn or time.sleep,
            metrics if metrics is not None else InMemoryMetrics(),
        )
        self.sharding = self._router.admin

    @functools.wraps(QdrantClient.query_points)
    def query_points(self, *args, **kwargs):
        flat = bind_call(QdrantClient.query_points, self, args, kwargs)
        return self._router.read(QdrantClient.query_points, "query_filter", flat)

    @functools.wraps(QdrantClient.query_points_groups)
    def query_points_groups(self, *args, **kwargs):
        flat = bind_call(QdrantClient.query_points_groups, self, args, kwargs)
        return self._router.read(QdrantClient.query_points_groups, "query_filter", flat)

    @functools.wraps(QdrantClient.search_matrix_pairs)
    def search_matrix_pairs(self, *args, **kwargs):
        flat = bind_call(QdrantClient.search_matrix_pairs, self, args, kwargs)
        return self._router.read(QdrantClient.search_matrix_pairs, "query_filter", flat)

    @functools.wraps(QdrantClient.search_matrix_offsets)
    def search_matrix_offsets(self, *args, **kwargs):
        flat = bind_call(QdrantClient.search_matrix_offsets, self, args, kwargs)
        return self._router.read(QdrantClient.search_matrix_offsets, "query_filter", flat)

    @functools.wraps(QdrantClient.count)
    def count(self, *args, **kwargs):
        flat = bind_call(QdrantClient.count, self, args, kwargs)
        return self._router.read(QdrantClient.count, "count_filter", flat)

    @functools.wraps(QdrantClient.scroll)
    def scroll(self, *args, **kwargs):
        flat = bind_call(QdrantClient.scroll, self, args, kwargs)
        return self._router.read(QdrantClient.scroll, "scroll_filter", flat)

    @functools.wraps(QdrantClient.facet)
    def facet(self, *args, **kwargs):
        flat = bind_call(QdrantClient.facet, self, args, kwargs)
        return self._router.read(QdrantClient.facet, "facet_filter", flat)

    @functools.wraps(QdrantClient.query_batch_points)
    def query_batch_points(self, *args, **kwargs):
        flat = bind_call(QdrantClient.query_batch_points, self, args, kwargs)
        return self._router.query_batch(flat)

    @functools.wraps(QdrantClient.upsert)
    def upsert(self, *args, **kwargs):
        flat = bind_call(QdrantClient.upsert, self, args, kwargs)
        return self._router.upsert(flat)

    @functools.wraps(QdrantClient.upload_points)
    def upload_points(self, *args, **kwargs):
        flat = bind_call(QdrantClient.upload_points, self, args, kwargs)
        return self._router.upload_points(flat)

    @functools.wraps(QdrantClient.upload_collection)
    def upload_collection(self, *args, **kwargs):
        flat = bind_call(QdrantClient.upload_collection, self, args, kwargs)
        return self._router.upload_collection(flat)

    @functools.wraps(QdrantClient.retrieve)
    def retrieve(self, *args, **kwargs):
        flat = bind_call(QdrantClient.retrieve, self, args, kwargs)
        return self._router.fanout(QdrantClient.retrieve, flat, write=False)

    @functools.wraps(QdrantClient.update_vectors)
    def update_vectors(self, *args, **kwargs):
        flat = bind_call(QdrantClient.update_vectors, self, args, kwargs)
        return self._router.selective(QdrantClient.update_vectors, "points", flat)

    @functools.wraps(QdrantClient.delete)
    def delete(self, *args, **kwargs):
        flat = bind_call(QdrantClient.delete, self, args, kwargs)
        return self._router.selective(QdrantClient.delete, "points_selector", flat)

    @functools.wraps(QdrantClient.set_payload)
    def set_payload(self, *args, **kwargs):
        flat = bind_call(QdrantClient.set_payload, self, args, kwargs)
        return self._router.payload(flat, overwrite=False)

    @functools.wraps(QdrantClient.overwrite_payload)
    def overwrite_payload(self, *args, **kwargs):
        flat = bind_call(QdrantClient.overwrite_payload, self, args, kwargs)
        return self._router.payload(flat, overwrite=True)

    @functools.wraps(QdrantClient.delete_vectors)
    def delete_vectors(self, *args, **kwargs):
        flat = bind_call(QdrantClient.delete_vectors, self, args, kwargs)
        return self._router.selective(QdrantClient.delete_vectors, "points", flat)

    @functools.wraps(QdrantClient.delete_payload)
    def delete_payload(self, *args, **kwargs):
        flat = bind_call(QdrantClient.delete_payload, self, args, kwargs)
        return self._router.delete_payload(flat)

    @functools.wraps(QdrantClient.clear_payload)
    def clear_payload(self, *args, **kwargs):
        flat = bind_call(QdrantClient.clear_payload, self, args, kwargs)
        return self._router.clear_payload(flat)

    @functools.wraps(QdrantClient.batch_update_points)
    def batch_update_points(self, *args, **kwargs):
        flat = bind_call(QdrantClient.batch_update_points, self, args, kwargs)
        return self._router.batch(flat)

    @functools.wraps(QdrantClient.delete_collection)
    def delete_collection(self, *args, **kwargs):
        flat = bind_call(QdrantClient.delete_collection, self, args, kwargs)
        return self._router.delete_collection(flat)


class _CallRouter:

    def __init__(
        self,
        client: ShardedQdrantClient,
        folder: MapFolder | None,
        ttl: float,
        sharding_warnings: bool,
        detect_feature_changes: bool,
        now_fn: Callable[[], datetime],
        sleep_fn: Callable[[int], None],
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._client = client
        self._folder = folder
        self._ttl = ttl
        self._sharding_warnings = sharding_warnings
        self._detect_feature_changes = detect_feature_changes
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._metrics = metrics if metrics is not None else InMemoryMetrics()
        self._caches: dict[str, MapCache] = {}
        self._methods: dict[str, str] = {}
        self._vectors = QdrantVectorStore(client)
        self.admin = ShardingAdmin(
            folder,
            self._cache_for,
            now_fn,
            sleep_fn,
            self._vectors,
            self._metrics,
        )

    def _invoke(self, method, flat: dict) -> Any:
        return method(self._client, **flat)

    def read(self, method, filter_name: str, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=False)
        if doc is None:
            return self._invoke(method, flat)
        doc = self._reload_if_unknown(collection, doc, flat.get(filter_name))
        if doc is None:
            return self._invoke(method, flat)
        slots = read_selector(doc, flat.get(filter_name))
        if slots is None:
            self._metrics.record_read(
                collection, method.__name__, fanout=True, slot_count=len(doc.slots)
            )
            warn(FanOutWarning, READ_FANOUT, self._sharding_warnings)
            return self._invoke(method, flat)
        self._metrics.record_read(
            collection, method.__name__, fanout=False, slot_count=len(slots)
        )
        flat["shard_key_selector"] = slots
        return self._invoke(method, flat)

    def query_batch(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, None, write=False)
        if doc is None:
            return self._invoke(QdrantClient.query_batch_points, flat)
        requests = []
        reloaded = False
        for request in flat["requests"]:
            if getattr(request, "shard_key", None) is not None:
                requests.append(request)
                continue
            if not reloaded and needs_reload(doc, getattr(request, "filter", None)):
                doc = self._reload_if_unknown(collection, doc, request.filter)
                reloaded = True
            if doc is None:
                requests.append(request)
                continue
            copied = request_on_slots(doc, request)
            if copied is request:
                self._metrics.record_read(
                    collection,
                    "query_batch_points",
                    fanout=True,
                    slot_count=len(doc.slots),
                )
                warn(FanOutWarning, READ_FANOUT, self._sharding_warnings)
            else:
                slot_list = getattr(copied, "shard_key", None) or []
                self._metrics.record_read(
                    collection,
                    "query_batch_points",
                    fanout=False,
                    slot_count=len(slot_list),
                )
            requests.append(copied)
        flat["requests"] = requests
        return self._invoke(QdrantClient.query_batch_points, flat)

    def upsert(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(QdrantClient.upsert, flat)
        result, moved, _doc = self._upsert_doc(
            flat, doc, check_moves=self._detect_feature_changes
        )
        if moved:
            self._metrics.record_feature_move(collection, moved)
            warn(
                FeatureMovedWarning,
                f"moved {moved} points",
                self._sharding_warnings,
            )
        return result

    def upload_points(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(QdrantClient.upload_points, flat)
        return self._send_chunks(flat, flat["points"])

    def upload_collection(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(QdrantClient.upload_collection, flat)
        points = iter_upload_points(flat["vectors"], flat.get("payload"), flat.get("ids"))
        return self._send_chunks(flat, points)

    def _send_chunks(self, flat: dict, points):
        if flat["parallel"] > 1:
            warn(ShardingWarning, PARALLEL_UPLOAD, self._sharding_warnings)
        for batch in chunks(points, flat["batch_size"]):
            self._client.upsert(
                **supported_kwargs(QdrantClient.upsert, upload_upsert(flat, batch))
            )
        return None

    def fanout(self, method, flat: dict, write: bool):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=write)
        if doc is None:
            return self._invoke(method, flat)
        if write:
            self._metrics.record_write(
                collection, method.__name__, fanout=True, slot_count=len(doc.slots)
            )
        else:
            self._metrics.record_read(
                collection, method.__name__, fanout=True, slot_count=len(doc.slots)
            )
        warn(FanOutWarning, CALL_FANOUT, self._sharding_warnings)
        return self._invoke(method, flat)

    def selective(self, method, selector_name: str, flat: dict, stacklevel: int = 4):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(method, flat)
        selector = flat.get(selector_name)
        if needs_reload(doc, selector_filter(selector)):
            self._cache_for(collection).expire()
            fresh = self.map_for(collection, None, write=True)
            if fresh is not None:
                doc = fresh
        if is_partial_update(method.__name__):
            self._copy_filter_matches(collection, doc, selector)
        slots = mutation_slots(doc, selector)
        if slots is not None:
            self._metrics.record_write(
                collection, method.__name__, fanout=False, slot_count=len(slots)
            )
            flat["shard_key_selector"] = slots
            return self._invoke(method, flat)
        self._metrics.record_write(
            collection, method.__name__, fanout=True, slot_count=len(doc.slots)
        )
        warn(FanOutWarning, SELECTOR_FANOUT, self._sharding_warnings, stacklevel=stacklevel)
        if write_uses_all_slots(method.__name__, selector):
            flat["shard_key_selector"] = list(doc.slots)
            return self._invoke(method, flat)
        return self._write_known_ids(method, selector_name, flat, doc)

    def _copy_filter_matches(self, collection: str, doc: MapDoc, selector: object) -> None:
        records = []
        for target in filter_copy_targets(doc, selector):
            for slot in target.slots:
                records.extend(
                    self._scroll_ids(collection, target.query, slot, doc.feature_field)
                )
        self._copy_records(collection, doc, records)

    def _scroll_ids(self, collection: str, query, slot: str, field: str):
        found = []
        offset = None
        while True:
            records, offset = self._invoke(
                QdrantClient.scroll,
                supported_kwargs(
                    QdrantClient.scroll,
                    id_scroll_page(collection, query, slot, field, offset),
                ),
            )
            found.extend(records)
            if offset is None:
                return found

    def _copy_records(self, collection: str, doc: MapDoc, records) -> set:
        kept = {record.id for record in records}
        for job in copy_on_write(doc, records):
            still = services.align_ids(
                self._vectors, collection, job.old_slot, job.new_slot, job.ids
            )
            for point_id in job.ids:
                if point_id not in still:
                    kept.discard(point_id)
        return kept

    def _write_known_ids(self, method, selector_name: str, flat: dict, doc: MapDoc):
        selector = flat.get(selector_name)
        ids = selector_ids(selector)
        collection = flat["collection_name"]
        records = self._invoke(
            QdrantClient.retrieve,
            supported_kwargs(
                QdrantClient.retrieve,
                feature_retrieve(collection, ids, doc.feature_field),
            ),
        )
        if features_need_reload(doc, records):
            self._cache_for(collection).expire()
            fresh = self.map_for(collection, None, write=True)
            if fresh is not None:
                doc = fresh
        kept = {record.id for record in records}
        if is_partial_update(method.__name__):
            kept = self._copy_records(collection, doc, records)
        plan = id_update_plan(doc, ids, records, kept)
        return self._send_id_plan(method, selector_name, flat, selector, plan)

    def _id_call(self, flat: dict, selector_name: str, selector, ids: list, slots: list[str]):
        call = dict(flat)
        call[selector_name] = selector_with_ids(selector, ids)
        call["shard_key_selector"] = slots
        return call

    def _send_id_plan(self, method, selector_name: str, flat: dict, selector, plan):
        result = None
        for group in plan.groups:
            result = self._invoke(
                method,
                self._id_call(flat, selector_name, selector, group.ids, group.slots),
            )
        if plan.probe_ids:
            wrote = False
            last = None
            for slot in plan.probe_slots:
                try:
                    result = self._invoke(
                        method,
                        self._id_call(flat, selector_name, selector, plan.probe_ids, [slot]),
                    )
                    wrote = True
                except UnexpectedResponse as exc:
                    if exc.status_code != 404:
                        raise
                    last = exc
            if not wrote and last is not None:
                raise last
        if plan.missing_ids and plan.missing_slot is not None:
            result = self._invoke(
                method,
                self._id_call(
                    flat, selector_name, selector, plan.missing_ids, [plan.missing_slot]
                ),
            )
        if result is None:
            return UpdateResult(status=UpdateStatus.COMPLETED)
        return result

    def payload(self, flat: dict, overwrite: bool):
        method = QdrantClient.overwrite_payload if overwrite else QdrantClient.set_payload
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(method, flat)
        if payload_moves_feature(doc, flat.get("payload"), flat.get("key"), overwrite):
            result, moved = self._move_payload(doc, flat, overwrite)
            if moved:
                self._metrics.record_feature_move(collection, moved)
                warn(
                    FeatureMovedWarning,
                    f"moved {moved} points",
                    self._sharding_warnings,
                )
            return result
        return self.selective(method, "points", flat, stacklevel=5)

    def delete_payload(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is not None:
            reject_feature_delete(doc, flat["keys"])
        return self.selective(QdrantClient.delete_payload, "points", flat, stacklevel=5)

    def clear_payload(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, flat.get("shard_key_selector"), write=True)
        if doc is None:
            return self._invoke(QdrantClient.clear_payload, flat)
        reject_clear(doc.feature_field)

    def batch(self, flat: dict):
        collection = flat["collection_name"]
        doc = self.map_for(collection, None, write=True)
        if doc is None:
            return self._invoke(QdrantClient.batch_update_points, flat)
        return [self._operation(flat, operation) for operation in flat["update_operations"]]

    def delete_collection(self, flat: dict):
        result = self._invoke(QdrantClient.delete_collection, flat)
        collection = flat["collection_name"]
        if self._folder is not None:
            self._folder.store_for(collection).delete()
        cache = self._caches.pop(collection, None)
        if cache is not None:
            cache.forget()
        self._methods.pop(collection, None)
        return result

    def map_for(self, collection: str, selector: object, write: bool) -> MapDoc | None:
        if selector is not None or self._folder is None:
            return None
        doc, reason = self._cache_for(collection).get()
        if reason is Reason.no_map:
            self._metrics.record_cache_event(collection, "unavailable")
            if not self._is_custom(collection):
                return None
            if write:
                self._metrics.record_write_denied(collection)
                warn(
                    StaleMapWarning,
                    MAP_UNAVAILABLE,
                    self._sharding_warnings,
                    stacklevel=_MAP_STACKLEVEL,
                )
                raise WriteDenied(f"map for {collection} is unavailable")
            warn(
                StaleMapWarning,
                MAP_UNAVAILABLE_READ,
                self._sharding_warnings,
                stacklevel=_MAP_STACKLEVEL,
            )
            return None
        if doc is None:
            return None
        if reason is Reason.stale_map:
            self._metrics.record_cache_event(collection, "stale")
            warn(
                StaleMapWarning,
                STALE_USED,
                self._sharding_warnings,
                stacklevel=_MAP_STACKLEVEL,
            )
        else:
            self._metrics.record_cache_event(collection, "hit")
        return doc

    def _reload_if_unknown(
        self, collection: str, doc: MapDoc | None, flt: object
    ) -> MapDoc | None:
        if doc is None or not needs_reload(doc, flt):
            return doc
        self._cache_for(collection).expire()
        return self.map_for(collection, None, write=False)

    def _upsert_doc(self, flat: dict, doc: MapDoc, check_moves: bool):
        collection = flat["collection_name"]
        structs = point_structs(flat["points"])
        if not structs:
            return self._invoke(QdrantClient.upsert, flat), 0, doc
        grouped, doc = self._with_features(collection, doc, structs)
        planned = self._move_plan(collection, doc, structs) if check_moves else []
        if not grouped:
            moved = self._delete_plan(collection, planned, flat.get("ordering")) if planned else 0
            return self._invoke(QdrantClient.upsert, flat), moved, doc
        wait = upsert_wait(bool(planned), flat["wait"])
        self._metrics.record_write(
            collection, "upsert", fanout=False, slot_count=len(grouped)
        )
        result = None
        for slot, slot_points in grouped.items():
            result = self._invoke(
                QdrantClient.upsert, slot_upsert(flat, slot, slot_points, wait)
            )
        moved = self._delete_plan(collection, planned, flat.get("ordering")) if planned else 0
        return result, moved, doc

    def _with_features(self, collection: str, doc: MapDoc, structs):
        grouped, new_values = plan_writes(doc, structs)
        if not new_values:
            return grouped, doc
        doc = services.add_features(self._folder.store_for(collection), new_values)
        self._cache_for(collection).put(doc)
        grouped, new_values = plan_writes(doc, structs)
        if new_values:
            raise ValueError(f"feature values were not added: {new_values}")
        return grouped, doc

    def _move_payload(self, doc: MapDoc, flat: dict, overwrite: bool):
        collection = flat["collection_name"]
        records = self._records(collection, flat["points"])
        if records is None:
            method = QdrantClient.overwrite_payload if overwrite else QdrantClient.set_payload
            return self.selective(method, "points", flat)
        if not records:
            return UpdateResult(status=UpdateStatus.COMPLETED), 0
        if features_need_reload(doc, records):
            self._cache_for(collection).expire()
            fresh = self.map_for(collection, None, write=True)
            if fresh is not None:
                doc = fresh
        points = points_with_payload(records, flat.get("payload"), overwrite)
        result, _moved, doc = self._upsert_doc(
            supported_kwargs(
                QdrantClient.upsert,
                moved_upsert(collection, points, flat.get("ordering"), flat.get("timeout")),
            ),
            doc,
            check_moves=False,
        )
        return result, self._delete_changed(
            collection, doc, records, points, flat.get("ordering")
        )

    def _records(self, collection: str, selector: object):
        source = record_source(selector)
        if source is None:
            return None
        kind, value = source
        if kind == "filter":
            return self._scroll_all(collection, value)
        return self._invoke(
            QdrantClient.retrieve,
            supported_kwargs(QdrantClient.retrieve, full_retrieve(collection, value)),
        )

    def _scroll_all(self, collection: str, query):
        found = []
        offset = None
        while True:
            records, offset = self._invoke(
                QdrantClient.scroll,
                supported_kwargs(QdrantClient.scroll, scroll_page(collection, query, offset)),
            )
            found.extend(records)
            if offset is None:
                return found

    def _move_plan(self, collection: str, doc: MapDoc, points):
        structs = point_structs(points)
        if not structs:
            return []
        records = self._invoke(
            QdrantClient.retrieve,
            supported_kwargs(
                QdrantClient.retrieve,
                feature_retrieve(
                    collection, [point.id for point in structs], doc.feature_field
                ),
            ),
        )
        if features_need_reload(doc, records):
            self._cache_for(collection).expire()
            fresh = self.map_for(collection, None, write=True)
            if fresh is not None:
                doc = fresh
        return moved_points(doc, stored_payloads(records), structs)

    def _delete_plan(self, collection: str, planned, ordering) -> int:
        for slot, ids in ids_by_slot(planned).items():
            self._invoke(
                QdrantClient.delete,
                supported_kwargs(
                    QdrantClient.delete, delete_ids(collection, ids, slot, ordering)
                ),
            )
        return len(planned)

    def _delete_changed(self, collection, doc, records, points, ordering) -> int:
        ordered = points_for_records(records, points)
        return self._delete_plan(
            collection, moved_points(doc, stored_payloads(records), ordered), ordering
        )

    def _operation(self, flat: dict, operation):
        spec = operation_call(operation)
        method = getattr(QdrantClient, spec.name)
        call = getattr(self._client, spec.name)
        values = supported_kwargs(method, {**batch_fields(flat), **spec.values})
        return call(**values)

    def _cache_for(self, collection: str) -> MapCache:
        cache = self._caches.get(collection)
        if cache is None:
            if self._folder is None:
                raise ValueError("no map folder")
            cache = MapCache(
                self._folder.store_for(collection),
                self._ttl,
                self._now_fn,
            )
            self._caches[collection] = cache
        return cache

    def _is_custom(self, collection: str) -> bool:
        known = self._methods.get(collection)
        if known is not None:
            return known == "custom"
        info = self._invoke(QdrantClient.get_collection, {"collection_name": collection})
        custom = info.config.params.sharding_method == ShardingMethod.CUSTOM
        self._methods[collection] = "custom" if custom else "auto"
        return custom


class ShardingAdmin:

    def __init__(
        self,
        folder: MapFolder | None,
        cache_for: Callable[[str], MapCache],
        now_fn: Callable[[], datetime],
        sleep_fn: Callable[[int], None],
        vectors: QdrantVectorStore,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._folder = folder
        self._cache_for = cache_for
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._vectors = vectors
        self.metrics: MetricsRecorder = (
            metrics if metrics is not None else InMemoryMetrics()
        )

    def create_collection(
        self,
        collection_name: str,
        feature_field: str,
        feature_type: str = "keyword",
        n_slots: int = 48,
        feature_counts: dict[str, int] | None = None,
        **create_collection_kwargs: object,
    ) -> MapDoc:
        """Create a custom-sharded collection, its slots, the feature index, and the map."""
        require_sdk_sharding(create_collection_kwargs)
        doc = services.create_sharded_collection(
            self._store(collection_name),
            self._vectors,
            collection_name,
            feature_field,
            feature_type,
            n_slots=n_slots,
            feature_counts=feature_counts,
            **create_collection_kwargs,
        )
        self._cache_for(collection_name).put(doc)
        return doc

    def add_feature(self, collection: str, value: int | str):
        """Place value on the least loaded slot."""
        info = services.add_feature(self._store(collection), value)
        self._sync(collection)
        return info

    def add_slot(self, collection: str, slot: str) -> MapDoc:
        """Append slot to the map and create its shard key."""
        doc = services.add_slot(self._store(collection), self._vectors, slot)
        self._cache_for(collection).put(doc)
        return doc

    def migrate(self, collection: str, value: int | str, to_slot: str, grace_seconds: int = 60):
        """Move value onto to_slot. Waits grace_seconds before copy, switch, and delete."""
        services.migrate(
            self._store(collection),
            self._vectors,
            value,
            [to_slot],
            self._now_fn,
            self._sleep_fn,
            grace_seconds,
        )
        return self._sync(collection)

    def resume_move(
        self, collection: str, value: int | str, grace_seconds: int = 60
    ) -> MapDoc:
        """Continue a move from its current state through stable."""
        services.resume_move(
            self._store(collection),
            self._vectors,
            value,
            self._now_fn,
            self._sleep_fn,
            grace_seconds,
        )
        return self._sync(collection)

    def split(
        self,
        collection: str,
        value: int | str,
        partitions: int,
        grace_seconds: int = 60,
    ):
        """Spread value across partitions free slots."""
        store = self._store(collection)
        targets = services.split_targets(store.load(), value, partitions)
        services.migrate(
            store,
            self._vectors,
            value,
            targets,
            self._now_fn,
            self._sleep_fn,
            grace_seconds,
        )
        return self._sync(collection)

    def start_move(
        self,
        collection: str,
        value: int | str,
        targets: list[str],
        grace_seconds: int = 60,
    ) -> None:
        """stable -> copying."""
        services.start_move(
            self._store(collection),
            value,
            targets,
            self._now_fn(),
            grace_seconds,
        )
        self._sync(collection)

    def copy_feature(self, collection: str, value: int | str, batch: int = 256) -> int:
        """Copy points from the old slots onto write_to."""
        copied = services.copy_feature(
            self._store(collection), self._vectors, value, self._now_fn(), batch
        )
        self._sync(collection)
        return copied

    def open_dual(self, collection: str, value: int | str, grace_seconds: int = 60) -> None:
        """copying -> dual."""
        services.open_dual(
            self._store(collection),
            value,
            self._now_fn(),
            grace_seconds,
        )
        self._sync(collection)

    def cutover(self, collection: str, value: int | str, grace_seconds: int = 60):
        """dual -> cutover. Returns the old slots and the reconcile counts."""
        result = services.cutover(
            self._store(collection),
            self._vectors,
            value,
            self._now_fn(),
            grace_seconds,
        )
        self._sync(collection)
        return result

    def finish_move(self, collection: str, value: int | str) -> None:
        """Delete the old slots, then cutover -> stable."""
        services.finish_move(
            self._store(collection),
            self._vectors,
            value,
            self._now_fn(),
        )
        self._sync(collection)

    def alias_feature(
        self,
        collection: str,
        old: int | str,
        new: int | str,
        grace_seconds: int = 60,
    ) -> int:
        """Rename old to new, wait grace_seconds, then set stored payloads to new."""
        try:
            return services.alias_feature(
                self._store(collection),
                self._vectors,
                old,
                new,
                self._sleep_fn,
                grace_seconds,
            )
        finally:
            self._sync(collection)

    def drop_alias(self, collection: str, old: int | str) -> int:
        """Rewrite points that still use old, then remove old from aliases."""
        try:
            return services.drop_alias(self._store(collection), self._vectors, old)
        finally:
            self._sync(collection)

    def read_load(self, collection: str) -> LoadSnapshot:
        """Return per-slot feature counts from Qdrant."""
        return services.read_load(self._store(collection), self._vectors)

    def plan_rebalance(self, collection: str) -> RebalancePlan:
        """Return a same-width plan from the live counts."""
        return services.plan_rebalance(self._store(collection), self._vectors)

    def apply_rebalance(
        self, collection: str, plan: RebalancePlan, grace_seconds: int = 60
    ) -> MapDoc:
        """Apply the plan in order. Skip a feature already on the planned slots."""
        services.apply_rebalance(
            self._store(collection),
            self._vectors,
            plan,
            self._now_fn,
            self._sleep_fn,
            grace_seconds,
        )
        return self._sync(collection)

    def refresh_stats(self, collection: str) -> MapDoc:
        """Replace points_count and slot_load from Qdrant."""
        doc = services.refresh_stats(self._store(collection), self._vectors)
        self._cache_for(collection).put(doc)
        return doc

    def get_map(self, collection: str) -> MapDoc | None:
        """Return the stored map, or None when the collection has no map."""
        try:
            doc = self._store(collection).load()
        except MapNotFound:
            return None
        self._cache_for(collection).put(doc)
        return doc

    def list_collections(self) -> list[str]:
        """Return sorted collection names that have a map in this map folder."""
        if self._folder is None:
            raise ValueError("no map folder")
        return list_map_collections(self._folder)

    def stats(self, collection: str, live: bool = False) -> CollectionStats:
        """Compute distribution statistics from the stored map or live Qdrant counts."""
        doc = self._store(collection).load()
        snapshot = services.collect_load(self._vectors, doc) if live else None
        return collection_stats(doc, snapshot=snapshot, now=self._now_fn())

    def _store(self, collection: str):
        if self._folder is None:
            raise ValueError("no map folder")
        return self._folder.store_for(collection)

    def _sync(self, collection: str) -> MapDoc:
        doc = self._store(collection).load()
        self._cache_for(collection).put(doc)
        return doc
