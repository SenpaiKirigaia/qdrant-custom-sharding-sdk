import uuid
from dataclasses import dataclass

from qdrant_client.models import Batch, Filter, FilterSelector, PointStruct

from qdrant_custom_sharding.filters import feature_values
from qdrant_custom_sharding.models import FeatureState, MapDoc
from qdrant_custom_sharding.routing import read_slots, write_slots


def read_selector(doc: MapDoc, flt: object) -> list[str] | None:
    values = feature_values(flt, doc.feature_field)
    if not values:
        return None
    chosen: list[str] = []
    seen: set[str] = set()
    for value in values:
        found = doc.find(value)
        if found is None:
            return None
        for slot in read_slots(found[1]):
            if slot in seen:
                continue
            seen.add(slot)
            chosen.append(slot)
    return chosen


def plan_writes(
    doc: MapDoc,
    points: list[PointStruct] | Batch,
) -> tuple[dict[str, list[PointStruct]], list[int | str]]:
    grouped: dict[str, list[PointStruct]] = {}
    new_values: list[int | str] = []
    seen_new: set[str] = set()
    for point in point_structs(points):
        value = _feature_value(point, doc.feature_field, doc.feature_type)
        found = doc.find(value)
        if found is None:
            key = str(value)
            if key not in seen_new:
                seen_new.add(key)
                new_values.append(value)
            continue
        for slot in write_slots(found[1], point.id):
            grouped.setdefault(slot, []).append(point)
    return grouped, new_values


def needs_reload(doc: MapDoc, flt: object) -> bool:
    values = feature_values(flt, doc.feature_field)
    if not values:
        return False
    return any(doc.find(value) is None for value in values)


@dataclass(frozen=True)
class FilterCopy:

    slots: tuple[str, ...]
    query: Filter


def filter_copy_targets(doc: MapDoc, selector: object) -> list[FilterCopy]:
    query = selector_filter(selector)
    if query is None:
        return []
    values = feature_values(query, doc.feature_field)
    if values:
        chosen: list[FilterCopy] = []
        seen: set[str] = set()
        for value in values:
            found = doc.find(value)
            if found is None:
                continue
            name, info = found
            if info.state != FeatureState.copying or name in seen:
                continue
            seen.add(name)
            chosen.append(FilterCopy(tuple(info.shard_keys), query))
        return chosen
    return [
        FilterCopy(tuple(info.shard_keys), query)
        for info in doc.features.values()
        if info.state == FeatureState.copying
    ]


def mutation_slots(doc: MapDoc, selector: object) -> list[str] | None:
    query = selector_filter(selector)
    if query is None:
        return None
    values = feature_values(query, doc.feature_field)
    if not values:
        return None
    chosen: list[str] = []
    seen: set[str] = set()
    for value in values:
        found = doc.find(value)
        if found is None:
            return None
        for slot in _held_slots(found[1]):
            if slot in seen:
                continue
            seen.add(slot)
            chosen.append(slot)
    return chosen


def is_feature_move(
    doc: MapDoc, point_id: int | str, old_value: object, new_value: object
) -> bool:
    old = _as_feature(old_value)
    new = _as_feature(new_value)
    if old is None or new is None:
        return False
    old_found = doc.find(old)
    new_found = doc.find(new)
    if old_found is None or new_found is None:
        return False
    if old_found[0] != new_found[0]:
        return True
    return bool(slots_to_delete(doc, point_id, old, new))


def slots_to_delete(
    doc: MapDoc, point_id: int | str, old_value: object, new_value: object
) -> list[str]:
    old = _as_feature(old_value)
    new = _as_feature(new_value)
    if old is None or new is None:
        return []
    old_found = doc.find(old)
    new_found = doc.find(new)
    if old_found is None or new_found is None:
        return []
    keep = set(write_slots(new_found[1], point_id))
    return [slot for slot in write_slots(old_found[1], point_id) if slot not in keep]


def moved_points(
    doc: MapDoc,
    old_payloads: dict,
    new_points: list[PointStruct] | Batch,
) -> list[tuple[object, list[str]]]:
    planned: list[tuple[object, list[str]]] = []
    for point in point_structs(new_points):
        if point.id not in old_payloads:
            continue
        old = _payload_field(old_payloads[point.id], doc.feature_field)
        new = _payload_field(point.payload, doc.feature_field)
        if not is_feature_move(doc, point.id, old, new):
            continue
        planned.append((point.id, slots_to_delete(doc, point.id, old, new)))
    return planned


def chunks(items, size: int):
    if size == 0:
        raise ValueError("range() arg 3 must not be zero")
    if size < 0:
        return
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_upload_points(vectors, payload, ids):
    vectors = _sized_rows(vectors)
    payload = _sized_column(payload)
    ids = _sized_column(ids)
    count = _vector_count(vectors)
    if payload is not None and len(payload) != count:
        raise ValueError("vectors, payload, and ids must have the same length")
    if ids is not None and len(ids) != count:
        raise ValueError("vectors, payload, and ids must have the same length")
    return _point_rows(vectors, payload, ids, count)


def point_structs(points: list[PointStruct] | Batch) -> list[PointStruct]:
    if isinstance(points, Batch):
        return _from_batch(points)
    return list(points)


def _from_batch(batch: Batch) -> list[PointStruct]:
    payloads = batch.payloads
    vectors = batch.vectors
    points: list[PointStruct] = []
    for index, point_id in enumerate(batch.ids):
        payload = None if payloads is None else payloads[index]
        points.append(
            PointStruct(
                id=point_id,
                vector=_batch_vector(vectors, index),
                payload=payload,
            )
        )
    return points


def _batch_vector(vectors: object, index: int) -> object:
    if isinstance(vectors, dict):
        return {name: column[index] for name, column in vectors.items()}
    return vectors[index]


def selector_filter(selector: object) -> Filter | None:
    if isinstance(selector, Filter):
        return selector
    if isinstance(selector, FilterSelector):
        return selector.filter
    return None


def _payload_field(payload: object, field: str) -> object:
    if not payload:
        return None
    return dict(payload).get(field)


def _sized_rows(vectors):
    if isinstance(vectors, dict):
        return {name: _sized_column(column) for name, column in vectors.items()}
    if _indexable(vectors):
        return vectors
    return list(vectors)


def _sized_column(value):
    if value is None or _indexable(value):
        return value
    return list(value)


def _indexable(value) -> bool:
    if isinstance(value, dict):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _vector_count(vectors) -> int:
    if not isinstance(vectors, dict):
        return len(vectors)
    if not vectors:
        return 0
    width = len(next(iter(vectors.values())))
    for column in vectors.values():
        if len(column) < width:
            column[width - 1]
    return width


def _point_rows(vectors, payload, ids, count: int):
    for index in range(count):
        point_id = str(uuid.uuid4()) if ids is None else ids[index]
        body = {} if payload is None else payload[index]
        yield PointStruct(
            id=point_id,
            vector=_vector_at(vectors, index),
            payload=dict(body),
        )


def _vector_at(vectors, index: int):
    if isinstance(vectors, dict):
        return {name: _cell(column, index) for name, column in vectors.items()}
    return _cell(vectors, index)


def _cell(column, index: int):
    if hasattr(column, "shape") and hasattr(column, "tolist"):
        return column[index].tolist()
    return column[index]


def _held_slots(info) -> list[str]:
    slots = list(info.shard_keys)
    if info.write_to:
        slots.extend(info.write_to)
    return slots


def _as_feature(value: object) -> int | str | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return value


def _feature_value(
    point: PointStruct, field: str, feature_type: str = "keyword"
) -> int | str:
    payload = point.payload
    if payload is None or field not in payload:
        raise ValueError(f"point {point.id} has no {field}")
    value = payload[field]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"point {point.id} has invalid {field}: {value!r}")
    if feature_type == "integer" and isinstance(value, str):
        try:
            int(value)
        except ValueError as exc:
            raise ValueError(
                f"point {point.id} has non-integer {field}: {value!r}"
            ) from exc
    return value
