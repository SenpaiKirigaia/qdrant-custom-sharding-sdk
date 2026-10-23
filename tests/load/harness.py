from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from qdrant_custom_sharding import ShardedQdrantClient
from qdrant_custom_sharding.errors import InvalidMove, MapNotFound
from qdrant_custom_sharding.models import FeatureState

_VECTOR = [0.1, 0.2, 0.3, 0.4]
_UPSERT_BATCH = 256
_COUNTED = (
    UnexpectedResponse,
    ResponseHandlingException,
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)


@dataclass(frozen=True)
class LoadHarnessRequest:
    url: str
    map_uri: str
    collection: str
    feature_field: str
    feature: int | str
    point_count: int
    workers: int
    grace_seconds: int


@dataclass(frozen=True)
class LoadHarnessReport:
    http_errors: int
    client_timeouts: int


def run_load_harness(request: LoadHarnessRequest) -> LoadHarnessReport:
    if not request.map_uri.startswith(("s3://", "file:")):
        Path(request.map_uri).mkdir(parents=True, exist_ok=True)
    client = _open_client(request)
    try:
        if not client.collection_exists(collection_name=request.collection):
            client.sharding.create_collection(
                collection_name=request.collection,
                feature_field=request.feature_field,
                feature_type=_feature_type(request.feature),
                n_slots=2,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )
        _upsert(client, request)
        stop = threading.Event()
        query_filter = _feature_filter(request)
        with ThreadPoolExecutor(max_workers=request.workers) as pool:
            futures = [
                pool.submit(_worker, request, stop, query_filter)
                for _ in range(request.workers)
            ]
            try:
                _migrate(client, request)
            finally:
                stop.set()
            http_errors = 0
            client_timeouts = 0
            for future in futures:
                http_count, timeout_count = future.result()
                http_errors += http_count
                client_timeouts += timeout_count
        return LoadHarnessReport(
            http_errors=http_errors,
            client_timeouts=client_timeouts,
        )
    finally:
        client.close()


def _open_client(request: LoadHarnessRequest) -> ShardedQdrantClient:
    # qdrant-client disables keepalive for localhost. Each call then waits out
    # TIME_WAIT, and Windows runs out of ports (WinError 10048).
    kwargs: dict[str, object] = {
        "url": request.url,
        "map_uri": request.map_uri,
        "timeout": 60,
        "prefer_grpc": False,
        "trust_env": False,
        "limits": httpx.Limits(max_connections=8, max_keepalive_connections=8),
    }
    if "check_compatibility" in inspect.signature(QdrantClient.__init__).parameters:
        kwargs["check_compatibility"] = False
    return ShardedQdrantClient(**kwargs)


def _feature_type(feature: int | str) -> str:
    if isinstance(feature, int) and not isinstance(feature, bool):
        return "integer"
    return "keyword"


def _feature_filter(request: LoadHarnessRequest) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key=request.feature_field,
                match=MatchValue(value=request.feature),
            )
        ]
    )


def _upsert(client: ShardedQdrantClient, request: LoadHarnessRequest) -> None:
    start = 0
    while start < request.point_count:
        end = min(start + _UPSERT_BATCH, request.point_count)
        points = [
            PointStruct(
                id=point_id,
                vector=_VECTOR,
                payload={request.feature_field: request.feature},
            )
            for point_id in range(start, end)
        ]
        client.upsert(collection_name=request.collection, points=points, wait=True)
        start = end


def _migrate(client: ShardedQdrantClient, request: LoadHarnessRequest) -> None:
    doc = client.sharding.get_map(request.collection)
    if doc is None:
        raise MapNotFound("map not found")
    found = doc.find(request.feature)
    if found is None:
        raise InvalidMove(f"unknown feature {request.feature}")
    info = found[1]
    if info.state != FeatureState.stable:
        client.sharding.resume_move(
            collection=request.collection,
            value=request.feature,
            grace_seconds=request.grace_seconds,
        )
        return
    exclude = set(info.shard_keys)
    if info.write_to:
        exclude.update(info.write_to)
    exclude.update(doc.busy_slots())
    to_slot = doc.least_loaded(1, exclude)[0]
    client.sharding.migrate(
        collection=request.collection,
        value=request.feature,
        to_slot=to_slot,
        grace_seconds=request.grace_seconds,
    )


def _worker(
    request: LoadHarnessRequest,
    stop: threading.Event,
    query_filter: Filter,
) -> tuple[int, int]:
    http_errors = 0
    client_timeouts = 0
    reader = _open_client(request)

    def record(call) -> None:
        nonlocal http_errors, client_timeouts
        try:
            call()
        except _COUNTED as exc:
            kind = _failure_kind(exc)
            if kind == "timeout":
                client_timeouts += 1
            elif kind == "http":
                http_errors += 1
            else:
                raise

    try:
        while not stop.is_set():
            record(
                lambda: reader.query_points(
                    collection_name=request.collection,
                    query=_VECTOR,
                    query_filter=query_filter,
                    limit=10,
                )
            )
            record(
                lambda: reader.count(
                    collection_name=request.collection,
                    count_filter=query_filter,
                )
            )
        return http_errors, client_timeouts
    finally:
        reader.close()


def _failure_kind(exc: BaseException) -> str | None:
    if isinstance(exc, ResponseHandlingException):
        return _failure_kind(exc.source)
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(exc, UnexpectedResponse):
        return "http"
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return "http"
    return None


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    report = run_load_harness(
        LoadHarnessRequest(
            url="http://localhost:6333",
            map_uri=str(repo / "maps"),
            collection="load_harness",
            feature_field="feature",
            feature="load",
            point_count=50000,
            workers=8,
            grace_seconds=60,
        )
    )
    print(report.http_errors, report.client_timeouts)
