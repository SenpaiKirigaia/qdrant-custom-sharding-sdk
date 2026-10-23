import inspect
from dataclasses import dataclass

from qdrant_client.models import (
    ClearPayloadOperation,
    DeleteOperation,
    DeletePayloadOperation,
    DeleteVectorsOperation,
    FilterSelector,
    OverwritePayloadOperation,
    PointIdsList,
    PointStruct,
    PointsBatch,
    SetPayloadOperation,
    ShardingMethod,
    UpdateVectorsOperation,
    UpsertOperation,
)

from qdrant_custom_sharding.models import FeatureState, MapDoc
from qdrant_custom_sharding.router import moved_points, read_selector, selector_filter
from qdrant_custom_sharding.routing import pick_slot

READ_FANOUT = "filter does not select one feature, read fans out"
CALL_FANOUT = "call has no shard selector and fans out"
SELECTOR_FANOUT = "selector does not name one feature, call fans out"
PARALLEL_UPLOAD = "parallel upload runs one batch at a time"
STALE_USED = "using a stale map"
MAP_UNAVAILABLE = "map unavailable"
MAP_UNAVAILABLE_READ = "map unavailable, read fans out"

_SCROLL_PAGE = 64
_declared_names: dict[object, frozenset[str]] = {}


def bind_call(method: object, owner: object, args: tuple, kwargs: dict) -> dict:
    bound = inspect.signature(method).bind(owner, *args, **kwargs)
    bound.apply_defaults()
    flat = dict(bound.arguments)
    extra = flat.pop("kwargs", {})
    flat.pop("self", None)
    flat.update(extra)
    return flat


def supported_kwargs(method: object, kwargs: dict) -> dict:
    names = _declared_names.get(method)
    if names is None:
        names = frozenset(
            name
            for name, param in inspect.signature(method).parameters.items()
            if name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )
        _declared_names[method] = names
    return {key: value for key, value in kwargs.items() if key in names}


def require_sdk_sharding(kwargs: dict) -> None:
    if "sharding_method" in kwargs or "shard_number" in kwargs:
        raise ValueError("sharding_method and shard_number are set by the SDK")
    kwargs["sharding_method"] = ShardingMethod.CUSTOM
    kwargs["shard_number"] = 1


def request_on_slots(doc: MapDoc, request):
    slots = read_selector(doc, getattr(request, "filter", None))
    if slots is None or not hasattr(request, "model_copy"):
        return request
    return request.model_copy(update={"shard_key": slots})


def upsert_wait(moving: bool, wait: bool) -> bool:
    if moving:
        return True
    return wait


def upload_upsert(flat: dict, batch) -> dict:
    return {
        "collection_name": flat["collection_name"],
        "points": batch,
        "wait": flat.get("wait"),
        "update_filter": flat.get("update_filter"),
        "update_mode": flat.get("update_mode"),
    }


def slot_upsert(flat: dict, slot: str, points, wait: bool) -> dict:
    call = dict(flat)
    call["points"] = points
    call["shard_key_selector"] = slot
    call["wait"] = wait
    return call


def ids_by_slot(planned) -> dict[str, list]:
    by_slot: dict[str, list] = {}
    for point_id, slots in planned:
        for slot in slots:
            by_slot.setdefault(slot, []).append(point_id)
    return by_slot


def delete_ids(collection: str, ids: list, slot: str, ordering) -> dict:
    return {
        "collection_name": collection,
        "points_selector": ids,
        "shard_key_selector": slot,
        "wait": True,
        "ordering": ordering,
    }


def feature_retrieve(collection: str, ids: list, field: str) -> dict:
    return {
        "collection_name": collection,
        "ids": ids,
        "with_payload": [field],
        "with_vectors": False,
    }


def full_retrieve(collection: str, ids: list) -> dict:
    return {
        "collection_name": collection,
        "ids": ids,
        "with_payload": True,
        "with_vectors": True,
    }


_PARTIAL_UPDATES = frozenset(
    {
        "set_payload",
        "overwrite_payload",
        "delete_payload",
        "update_vectors",
        "delete_vectors",
    }
)


def is_partial_update(op: str) -> bool:
    return op in _PARTIAL_UPDATES


def id_scroll_page(collection: str, query, slot: str, field: str, offset) -> dict:
    return {
        "collection_name": collection,
        "scroll_filter": query,
        "shard_key_selector": slot,
        "with_payload": [field],
        "with_vectors": False,
        "limit": _SCROLL_PAGE,
        "offset": offset,
    }


def scroll_page(collection: str, query, offset) -> dict:
    return {
        "collection_name": collection,
        "scroll_filter": query,
        "with_payload": True,
        "with_vectors": True,
        "limit": _SCROLL_PAGE,
        "offset": offset,
    }


def moved_upsert(collection: str, points, ordering, timeout) -> dict:
    return {
        "collection_name": collection,
        "points": points,
        "wait": True,
        "ordering": ordering,
        "shard_key_selector": None,
        "update_filter": None,
        "update_mode": None,
        "timeout": timeout,
    }


def payload_moves_feature(doc: MapDoc, payload: object, key: object, overwrite: bool) -> bool:
    body = {} if not payload else dict(payload)
    if overwrite and doc.feature_field not in body:
        raise ValueError(f"overwrite_payload would remove {doc.feature_field}")
    return key is None and doc.feature_field in body


def reject_feature_delete(doc: MapDoc, keys) -> None:
    if doc.feature_field in list(keys):
        raise ValueError(f"delete_payload would remove {doc.feature_field}")


def reject_clear(field: str) -> None:
    raise ValueError(f"clear_payload would remove {field}")


def merged_payload(old: object, assigned: dict, overwrite: bool) -> dict:
    if overwrite:
        return dict(assigned)
    payload = {} if not old else dict(old)
    payload.update(assigned)
    return payload


def points_with_payload(records, assigned: object, overwrite: bool) -> list[PointStruct]:
    body = {} if not assigned else dict(assigned)
    return [
        PointStruct(
            id=record.id,
            vector=record.vector,
            payload=merged_payload(record.payload, body, overwrite),
        )
        for record in records
    ]


def stored_payloads(records) -> dict:
    return {record.id: record.payload for record in records}


def points_for_records(records, points) -> list:
    by_id = {point.id: point for point in points}
    return [by_id[record.id] for record in records if record.id in by_id]


def record_source(selector: object):
    query = selector_filter(selector)
    if query is not None:
        return "filter", query
    ids = ids_of(selector)
    if not ids:
        return None
    return "ids", ids


def ids_of(selector: object) -> list:
    if isinstance(selector, PointIdsList):
        return list(selector.points)
    if isinstance(selector, (list, tuple)):
        return list(selector)
    return []


def selector_value(selector: object):
    if isinstance(selector, FilterSelector):
        return selector.filter
    if isinstance(selector, PointIdsList):
        return list(selector.points)
    return selector


def selector_shard(selector: object):
    return getattr(selector, "shard_key", None)


def _is_point(value: object) -> bool:
    return hasattr(value, "id") and not isinstance(value, (int, str, bytes))


def selector_ids(selector: object) -> list:
    if isinstance(selector, PointIdsList):
        return list(selector.points)
    if isinstance(selector, (list, tuple)):
        if selector and _is_point(selector[0]):
            return [point.id for point in selector]
        return list(selector)
    return []


def selector_with_ids(selector: object, ids: list):
    if isinstance(selector, (list, tuple)) and selector and _is_point(selector[0]):
        chosen = set(ids)
        return [point for point in selector if point.id in chosen]
    return list(ids)


def write_uses_all_slots(op: str, selector: object) -> bool:
    """True when every shard key accepts this write.

    delete, and a filter, stay ok when a key has no matching point.
    Payload and vector updates by id return 404 in that case.
    """
    if op == "delete" or selector_filter(selector) is not None:
        return True
    return not selector_ids(selector)


def _record_slot(record) -> str | None:
    key = getattr(record, "shard_key", None)
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if key is None or key == "":
        return None
    return str(key)


def _field_value(record, field: str):
    payload = getattr(record, "payload", None)
    if not payload or field not in dict(payload):
        return None
    return dict(payload)[field]


def _point_slots(info, point_id) -> list[str]:
    old = pick_slot(info.shard_keys, point_id)
    if not info.write_to or info.state not in (
        FeatureState.copying,
        FeatureState.dual,
        FeatureState.cutover,
    ):
        return [old]
    new = pick_slot(info.write_to, point_id)
    if info.state == FeatureState.cutover:
        return [new]
    if new == old:
        return [old]
    return [old, new]


@dataclass(frozen=True)
class IdWriteGroup:

    ids: list
    slots: list[str]


@dataclass(frozen=True)
class CopyOnWrite:

    ids: list
    old_slot: str
    new_slot: str


def copy_on_write(doc: MapDoc, records) -> list[CopyOnWrite]:
    buckets: dict[tuple[str, str], list] = {}
    order: list[tuple[str, str]] = []
    seen: set[object] = set()
    for record in records:
        if record.id in seen:
            continue
        value = _field_value(record, doc.feature_field)
        if value is None:
            continue
        found = doc.find(value)
        if found is None:
            continue
        info = found[1]
        if info.state != FeatureState.copying or not info.write_to:
            continue
        seen.add(record.id)
        old = pick_slot(info.shard_keys, record.id)
        new = pick_slot(info.write_to, record.id)
        if old == new:
            continue
        key = (old, new)
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(record.id)
    return [CopyOnWrite(buckets[key], key[0], key[1]) for key in order]


def unknown_feature_ids(doc: MapDoc, records) -> list:
    ids = []
    seen: set[object] = set()
    for record in records:
        if record.id in seen:
            continue
        value = _field_value(record, doc.feature_field)
        if value is not None and doc.find(value) is not None:
            continue
        seen.add(record.id)
        ids.append(record.id)
    return ids


def features_need_reload(doc: MapDoc, records) -> bool:
    return bool(unknown_feature_ids(doc, records))


@dataclass(frozen=True)
class IdUpdatePlan:

    groups: list[IdWriteGroup]
    probe_ids: list
    probe_slots: list[str]
    missing_ids: list
    missing_slot: str | None


def id_update_plan(doc: MapDoc, wanted: list, records, kept) -> IdUpdatePlan:
    kept_ids = set(kept)
    routed = [record for record in records if record.id in kept_ids]
    groups = id_write_groups(doc, routed)
    placed = {point_id for group in groups for point_id in group.ids}
    probe: list = []
    seen_probe: set[object] = set()
    for record in routed:
        if record.id in placed or record.id in seen_probe:
            continue
        seen_probe.add(record.id)
        probe.append(record.id)
    found = {record.id for record in records}
    missing: list = []
    seen_missing: set[object] = set()
    for point_id in wanted:
        if point_id in found or point_id in seen_missing:
            continue
        seen_missing.add(point_id)
        missing.append(point_id)
    for record in records:
        if record.id in kept_ids or record.id in seen_missing:
            continue
        seen_missing.add(record.id)
        missing.append(record.id)
    slot = doc.slots[0] if doc.slots and missing else None
    return IdUpdatePlan(groups, probe, list(doc.slots), missing, slot)


def id_write_groups(doc: MapDoc, records) -> list[IdWriteGroup]:
    buckets: dict[str, list] = {}
    order: list[str] = []
    for record in records:
        value = _field_value(record, doc.feature_field)
        found = doc.find(value) if value is not None else None
        if found is None:
            held = _record_slot(record)
            targets = [held] if held is not None else []
        else:
            targets = _point_slots(found[1], record.id)
        for slot in targets:
            if slot not in buckets:
                order.append(slot)
                buckets[slot] = []
            if record.id not in buckets[slot]:
                buckets[slot].append(record.id)
    return [IdWriteGroup(buckets[slot], [slot]) for slot in order]


def batch_fields(flat: dict) -> dict:
    return {
        "collection_name": flat["collection_name"],
        "wait": flat.get("wait"),
        "ordering": flat.get("ordering"),
        "timeout": flat.get("timeout"),
    }


@dataclass(frozen=True)
class OperationCall:

    name: str
    values: dict


def operation_call(operation) -> OperationCall:
    if isinstance(operation, UpsertOperation):
        body = operation.upsert
        points = body.batch if isinstance(body, PointsBatch) else body.points
        return OperationCall(
            "upsert",
            {
                "points": points,
                "shard_key_selector": body.shard_key,
                "update_filter": getattr(body, "update_filter", None),
                "update_mode": getattr(body, "update_mode", None),
            },
        )
    if isinstance(operation, DeleteOperation):
        return OperationCall(
            "delete",
            {
                "points_selector": selector_value(operation.delete),
                "shard_key_selector": selector_shard(operation.delete),
            },
        )
    if isinstance(operation, SetPayloadOperation):
        body = operation.set_payload
        return OperationCall(
            "set_payload",
            {
                "payload": body.payload,
                "points": body.points if body.points is not None else body.filter,
                "key": getattr(body, "key", None),
                "shard_key_selector": body.shard_key,
            },
        )
    if isinstance(operation, OverwritePayloadOperation):
        body = operation.overwrite_payload
        return OperationCall(
            "overwrite_payload",
            {
                "payload": body.payload,
                "points": body.points if body.points is not None else body.filter,
                "shard_key_selector": body.shard_key,
            },
        )
    if isinstance(operation, DeletePayloadOperation):
        body = operation.delete_payload
        return OperationCall(
            "delete_payload",
            {
                "keys": body.keys,
                "points": body.points if body.points is not None else body.filter,
                "shard_key_selector": body.shard_key,
            },
        )
    if isinstance(operation, ClearPayloadOperation):
        return OperationCall(
            "clear_payload",
            {
                "points_selector": selector_value(operation.clear_payload),
                "shard_key_selector": selector_shard(operation.clear_payload),
            },
        )
    if isinstance(operation, UpdateVectorsOperation):
        body = operation.update_vectors
        return OperationCall(
            "update_vectors",
            {
                "points": body.points,
                "shard_key_selector": body.shard_key,
                "update_filter": getattr(body, "update_filter", None),
            },
        )
    if isinstance(operation, DeleteVectorsOperation):
        body = operation.delete_vectors
        points = body.points if body.points is not None else body.filter
        return OperationCall(
            "delete_vectors",
            {
                "vectors": body.vector,
                "points": points,
                "shard_key_selector": body.shard_key,
            },
        )
    raise ValueError(f"unsupported update operation {type(operation).__name__}")
