import json
import socket
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PointVectors,
    ShardingMethod,
    VectorParams,
)

from tests.local_qdrant import local_kwargs

pytestmark = pytest.mark.integration

KEYS = ["s0", "s1", "s2"]
POINT_ID = 1
VECTOR = [0.1, 0.2, 0.3, 0.4]
NEW_VECTOR = [0.4, 0.3, 0.2, 0.1]
OPS = (
    "delete ids",
    "set_payload ids",
    "overwrite_payload ids",
    "delete_payload ids",
    "clear_payload ids",
    "update_vectors",
    "delete_vectors ids",
    "delete filter",
    "set_payload filter",
    "overwrite_payload filter",
    "delete_payload filter",
    "clear_payload filter",
    "delete_vectors filter",
)

CUSTOM = {
    "delete ids a": "400|tag=red,vec=same",
    "delete ids b": "ok|absent",
    "delete ids c": "ok|absent",
    "set_payload ids a": "400|tag=red,vec=same",
    "set_payload ids b": "404|tag=blue,vec=same",
    "set_payload ids c": "ok|tag=blue,vec=same",
    "overwrite_payload ids a": "400|tag=red,vec=same",
    "overwrite_payload ids b": "404|tag=green,vec=same",
    "overwrite_payload ids c": "ok|tag=green,vec=same",
    "delete_payload ids a": "400|tag=red,vec=same",
    "delete_payload ids b": "404|tag=None,vec=same",
    "delete_payload ids c": "ok|tag=None,vec=same",
    "clear_payload ids a": "400|tag=red,vec=same",
    "clear_payload ids b": "404|tag=None,vec=same",
    "clear_payload ids c": "ok|tag=None,vec=same",
    "update_vectors a": "400|tag=red,vec=same",
    "update_vectors b": "404|tag=red,vec=changed",
    "update_vectors c": "ok|tag=red,vec=changed",
    "delete_vectors ids a": "400|tag=red,vec=same",
    "delete_vectors ids b": "404|tag=red,vec=none",
    "delete_vectors ids c": "ok|tag=red,vec=none",
    "delete filter a": "400|tag=red,vec=same",
    "delete filter b": "ok|absent",
    "delete filter c": "ok|absent",
    "set_payload filter a": "400|tag=red,vec=same",
    "set_payload filter b": "ok|tag=blue,vec=same",
    "set_payload filter c": "ok|tag=blue,vec=same",
    "overwrite_payload filter a": "400|tag=red,vec=same",
    "overwrite_payload filter b": "ok|tag=green,vec=same",
    "overwrite_payload filter c": "ok|tag=green,vec=same",
    "delete_payload filter a": "400|tag=red,vec=same",
    "delete_payload filter b": "ok|tag=None,vec=same",
    "delete_payload filter c": "ok|tag=None,vec=same",
    "clear_payload filter a": "400|tag=red,vec=same",
    "clear_payload filter b": "ok|tag=None,vec=same",
    "clear_payload filter c": "ok|tag=None,vec=same",
    "delete_vectors filter a": "400|tag=red,vec=same",
    "delete_vectors filter b": "ok|tag=red,vec=none",
    "delete_vectors filter c": "ok|tag=red,vec=none",
    "delete missing all": "ok|tag=red,vec=same",
    "set_payload missing all": "404|tag=red,vec=same",
}

# One request with every shard key runs the keys together and returns on the first
# error. A 404 can arrive before the slot that holds the point is updated.
EITHER = {
    "set_payload ids b": {"404|tag=blue,vec=same", "404|tag=red,vec=same"},
    "overwrite_payload ids b": {"404|tag=green,vec=same", "404|tag=red,vec=same"},
    "delete_payload ids b": {"404|tag=None,vec=same", "404|tag=red,vec=same"},
    "clear_payload ids b": {"404|tag=None,vec=same", "404|tag=red,vec=same"},
    "update_vectors b": {"404|tag=red,vec=changed", "404|tag=red,vec=same"},
    "delete_vectors ids b": {"404|tag=red,vec=none", "404|tag=red,vec=same"},
}

AUTO = {
    "delete ids": "ok|other=blue",
    "set_payload ids": "404|other=blue",
    "overwrite_payload ids": "404|other=blue",
    "delete_payload ids": "404|other=blue",
    "clear_payload ids": "404|other=blue",
    "update_vectors": "404|other=blue",
    "delete_vectors ids": "404|other=blue",
    "delete filter": "ok|other=blue",
    "set_payload filter": "ok|other=blue",
    "set_payload mixed": "404|other=green",
    "delete mixed": "ok|other=absent",
}


def _qdrant_up() -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 6333))
    except OSError:
        return False
    finally:
        sock.close()
    return True


@pytest.fixture(scope="module", autouse=True)
def qdrant_up():
    if not _qdrant_up():
        pytest.skip("Qdrant is not running on localhost:6333")


def _error_text(exc: UnexpectedResponse) -> str:
    raw = exc.content.decode() if isinstance(exc.content, bytes) else str(exc.content)
    error = json.loads(raw)["status"]["error"]
    return error if isinstance(error, str) else str(error)


def _status(fn) -> str:
    try:
        fn()
    except UnexpectedResponse as exc:
        text = _error_text(exc)
        if exc.status_code == 400 and "Shard key not specified" in text:
            return "400"
        if exc.status_code == 404 and "No point with id" in text:
            return "404"
        return f"{exc.status_code} {text}"
    return "ok"


def _tag_filter() -> Filter:
    return Filter(must=FieldCondition(key="tag", match=MatchValue(value="red")))


def _apply(client: QdrantClient, name: str, op: str, selector, point_id: int = POINT_ID):
    extra = {"wait": True}
    if selector is not None:
        extra["shard_key_selector"] = selector
    query = _tag_filter()
    if op == "delete ids":
        client.delete(name, [point_id], **extra)
    elif op == "set_payload ids":
        client.set_payload(name, {"tag": "blue"}, [point_id], **extra)
    elif op == "overwrite_payload ids":
        client.overwrite_payload(
            name, {"feature": "alpha", "tag": "green"}, [point_id], **extra
        )
    elif op == "delete_payload ids":
        client.delete_payload(name, ["tag"], [point_id], **extra)
    elif op == "clear_payload ids":
        client.clear_payload(name, [point_id], **extra)
    elif op == "update_vectors":
        client.update_vectors(
            name, [PointVectors(id=point_id, vector=NEW_VECTOR)], **extra
        )
    elif op == "delete_vectors ids":
        client.delete_vectors(name, [""], [point_id], **extra)
    elif op == "delete filter":
        client.delete(name, query, **extra)
    elif op == "set_payload filter":
        client.set_payload(name, {"tag": "blue"}, query, **extra)
    elif op == "overwrite_payload filter":
        client.overwrite_payload(
            name, {"feature": "alpha", "tag": "green"}, query, **extra
        )
    elif op == "delete_payload filter":
        client.delete_payload(name, ["tag"], query, **extra)
    elif op == "clear_payload filter":
        client.clear_payload(name, query, **extra)
    elif op == "delete_vectors filter":
        client.delete_vectors(name, [""], query, **extra)
    else:
        raise AssertionError(op)


def _numbers(vector) -> list[float] | None:
    if isinstance(vector, dict):
        vector = vector.get("") if "" in vector else next(iter(vector.values()), None)
    if not vector:
        return None
    return [float(item) for item in vector]


def _same(left, right) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    return all(abs(item - other) < 1e-4 for item, other in zip(left, right))


def _effect(client: QdrantClient, name: str, keys: list[str], baseline) -> str:
    for key in keys:
        rows = client.retrieve(
            name,
            [POINT_ID],
            with_payload=True,
            with_vectors=True,
            shard_key_selector=key,
        )
        if not rows:
            continue
        payload = dict(rows[0].payload or {})
        nums = _numbers(rows[0].vector)
        if nums is None:
            vec = "none"
        elif _same(nums, baseline):
            vec = "same"
        else:
            vec = "changed"
        return f"tag={payload.get('tag')},vec={vec}"
    return "absent"


def _reset(client: QdrantClient, name: str) -> None:
    client.upsert(
        name,
        [
            PointStruct(
                id=POINT_ID,
                vector=VECTOR,
                payload={"feature": "alpha", "tag": "red"},
            )
        ],
        shard_key_selector=KEYS[0],
        wait=True,
    )
    for key in KEYS[1:]:
        rows = client.retrieve(name, [POINT_ID], shard_key_selector=key)
        if rows:
            client.delete(name, [POINT_ID], shard_key_selector=key, wait=True)


def _make_custom(client: QdrantClient, name: str) -> None:
    client.create_collection(
        name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        sharding_method=ShardingMethod.CUSTOM,
        shard_number=1,
    )
    for key in KEYS:
        client.create_shard_key(name, key)


def _make_auto(client: QdrantClient, name: str) -> None:
    client.create_collection(
        name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    client.upsert(
        name,
        [PointStruct(id=2, vector=VECTOR, payload={"feature": "alpha", "tag": "blue"})],
        wait=True,
    )


def _other_tag(client: QdrantClient, name: str) -> str:
    rows = client.retrieve(name, [2], with_payload=True)
    if not rows:
        return "absent"
    return str(dict(rows[0].payload or {}).get("tag"))


def test_custom_shard_write_rules():
    client = QdrantClient(**local_kwargs())
    name = "w" + uuid.uuid4().hex[:12]
    try:
        _make_custom(client, name)
        _reset(client, name)
        baseline = _numbers(
            client.retrieve(
                name, [POINT_ID], with_vectors=True, shard_key_selector=KEYS[0]
            )[0].vector
        )
        seen = {}
        modes = (("a", None), ("b", list(KEYS)), ("c", KEYS[0]))
        for op in OPS:
            for mode, selector in modes:
                _reset(client, name)
                status = _status(
                    lambda op=op, selector=selector: _apply(client, name, op, selector)
                )
                seen[f"{op} {mode}"] = f"{status}|{_effect(client, name, KEYS, baseline)}"
        _reset(client, name)
        seen["delete missing all"] = (
            _status(lambda: client.delete(name, [9], shard_key_selector=list(KEYS), wait=True))
            + "|"
            + _effect(client, name, KEYS, baseline)
        )
        _reset(client, name)
        seen["set_payload missing all"] = (
            _status(
                lambda: client.set_payload(
                    name, {"tag": "blue"}, [9], shard_key_selector=list(KEYS), wait=True
                )
            )
            + "|"
            + _effect(client, name, KEYS, baseline)
        )
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()
    assert set(seen) == set(CUSTOM)
    for key, expected in CUSTOM.items():
        if key in EITHER:
            assert seen[key] in EITHER[key]
        else:
            assert seen[key] == expected


def test_auto_shard_missing_ids():
    client = QdrantClient(**local_kwargs())
    name = "a" + uuid.uuid4().hex[:12]
    try:
        _make_auto(client, name)
        seen = {}
        for op in (
            "delete ids",
            "set_payload ids",
            "overwrite_payload ids",
            "delete_payload ids",
            "clear_payload ids",
            "update_vectors",
            "delete_vectors ids",
            "delete filter",
            "set_payload filter",
        ):
            status = _status(lambda op=op: _apply(client, name, op, None))
            seen[op] = f"{status}|other={_other_tag(client, name)}"
        status = _status(lambda: client.set_payload(name, {"tag": "green"}, [2, 1], wait=True))
        seen["set_payload mixed"] = f"{status}|other={_other_tag(client, name)}"
        status = _status(lambda: client.delete(name, [2, 1], wait=True))
        seen["delete mixed"] = f"{status}|other={_other_tag(client, name)}"
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()
    assert seen == AUTO
