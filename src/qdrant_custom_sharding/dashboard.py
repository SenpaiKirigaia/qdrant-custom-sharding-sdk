from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import grpc
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from qdrant_custom_sharding import services
from qdrant_custom_sharding.dashboard_ui import DASHBOARD_HTML
from qdrant_custom_sharding.errors import (
    InvalidMove,
    MapNotFound,
    MapUnavailable,
    MoveNotReady,
    ShardingError,
    VersionConflict,
)
from qdrant_custom_sharding.map_codec import map_to_json
from qdrant_custom_sharding.map_stores import list_map_collections, open_map_folder
from qdrant_custom_sharding.metrics import MetricsRecorder, render_prometheus
from qdrant_custom_sharding.models import MapDoc
from qdrant_custom_sharding.planning import plan_rebalance as build_rebalance_plan
from qdrant_custom_sharding.ports import MapFolder, VectorStore
from qdrant_custom_sharding.routing import pick_slot, read_slots, write_slots
from qdrant_custom_sharding.sharded_client import ShardedQdrantClient
from qdrant_custom_sharding.stats import collection_stats, recommend_splits

_MAX_BODY_BYTES = 1_048_576
_MAX_JOBS = 100


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    text = dt.astimezone(timezone.utc).isoformat()
    return text[:-6] + "Z" if text.endswith("+00:00") else text


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class JobRecord:
    id: str
    collection: str
    action: str
    feature: str | None
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "action": self.action,
            "feature": self.feature,
            "status": self.status.value,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """Runs long-lived admin operations on background threads with status tracking."""

    def __init__(self, now_fn: Callable[[], datetime] = _now) -> None:
        self._now = now_fn
        self._lock = threading.Lock()
        self._jobs: list[JobRecord] = []

    def submit(
        self,
        collection: str,
        action: str,
        feature: str | None,
        task: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        record = JobRecord(
            id=uuid.uuid4().hex,
            collection=collection,
            action=action,
            feature=feature,
            status=JobStatus.queued,
            created_at=self._now(),
        )
        with self._lock:
            self._jobs.insert(0, record)
            if len(self._jobs) > _MAX_JOBS:
                del self._jobs[_MAX_JOBS:]

        def run() -> None:
            with self._lock:
                record.status = JobStatus.running
                record.started_at = self._now()
            try:
                output = task()
            except Exception as exc:
                with self._lock:
                    record.status = JobStatus.failed
                    record.error = str(exc)
                    record.finished_at = self._now()
                return
            with self._lock:
                record.status = JobStatus.completed
                record.result = output
                record.finished_at = self._now()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return record.to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs]


class DashboardController:
    """Framework-agnostic controller for sharding statistics, metrics, and management."""

    def __init__(
        self,
        folder: MapFolder,
        vectors: VectorStore | None = None,
        metrics: MetricsRecorder | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[int], None] | None = None,
        read_only: bool = False,
    ) -> None:
        self._folder = folder
        self._vectors = vectors
        self._metrics = metrics
        self._now = now_fn or _now
        self._sleep = sleep_fn or time.sleep
        self.read_only = read_only
        self._jobs = JobRunner(self._now)

    @classmethod
    def from_client(
        cls,
        client: ShardedQdrantClient,
        read_only: bool = False,
    ) -> DashboardController:
        router: Any = getattr(client, "_router", None)
        folder = router._folder if router is not None else None
        if folder is None:
            raise ValueError("client has no map folder")
        return cls(
            folder=folder,
            vectors=router._vectors,
            metrics=client.sharding.metrics,
            now_fn=router._now_fn,
            sleep_fn=router._sleep_fn,
            read_only=read_only,
        )

    @property
    def has_vector_store(self) -> bool:
        return self._vectors is not None

    def health(self) -> dict[str, Any]:
        collections = self._discover_collections()
        return {
            "status": "ok",
            "read_only": self.read_only,
            "has_vector_store": self.has_vector_store,
            "collection_count": len(collections),
        }

    def overview(self, live: bool = False) -> dict[str, Any]:
        items = self._all_collection_stats(live=live)
        summaries = [
            {
                "collection": item.collection,
                "feature_field": item.feature_field,
                "feature_type": item.feature_type,
                "version": item.version,
                "source": item.source,
                "slot_count": item.slot_count,
                "feature_count": item.feature_count,
                "total_points": item.total_points,
                "imbalance_ratio": item.imbalance_ratio,
                "gini_coefficient": item.gini_coefficient,
                "needs_rebalance": item.needs_rebalance,
                "active_moves": len(item.active_moves),
            }
            for item in items
        ]
        routing = self._metrics.snapshot().to_dict() if self._metrics is not None else None
        return {
            "collection_count": len(summaries),
            "collections": summaries,
            "routing": routing,
        }

    def collection_detail(self, collection: str, live: bool = False) -> dict[str, Any]:
        doc = self._store(collection).load()
        snapshot = (
            services.collect_load(self._require_vectors(), doc)
            if live and self._vectors is not None
            else None
        )
        stats = collection_stats(doc, snapshot=snapshot, now=self._now())
        return {
            "stats": stats.to_dict(),
            "map": _map_dict(doc),
        }

    def get_map_dict(self, collection: str) -> dict[str, Any]:
        return _map_dict(self._store(collection).load())

    def preview_rebalance(self, collection: str) -> dict[str, Any]:
        vectors = self._require_vectors()
        doc = self._store(collection).load()
        snapshot = services.collect_load(vectors, doc)
        plan = build_rebalance_plan(snapshot, doc)
        totals_before = {
            slot: sum(snapshot.counts.get(slot, {}).values()) for slot in doc.slots
        }
        totals_after = _simulate_plan_loads(doc, snapshot, plan)
        splits = recommend_splits(doc, snapshot=snapshot)
        return {
            "collection": plan.collection,
            "reason": plan.reason.value,
            "moves": [
                {
                    "feature": order.feature,
                    "to_slots": list(order.to_slots),
                    "points": order.points,
                }
                for order in plan.moves
            ],
            "loads_before": totals_before,
            "loads_after": totals_after,
            "max_before": max(totals_before.values(), default=0),
            "max_after": max(totals_after.values(), default=0),
            "split_recommendations": [item.to_dict() for item in splits],
        }

    def inspect_route(
        self,
        collection: str,
        feature: int | str,
        point_id: int | str | None = None,
    ) -> dict[str, Any]:
        doc = self._store(collection).load()
        found = doc.find(feature)
        if found is None:
            return {
                "collection": collection,
                "feature": str(feature),
                "known": False,
                "fanout": True,
                "read_slots": list(doc.slots),
                "write_slots": None,
                "picked_slot": None,
            }
        canonical, info = found
        r_slots = read_slots(info)
        if point_id is not None:
            w_slots = write_slots(info, point_id)
            picked = pick_slot(r_slots, point_id)
        else:
            w_slots = list(info.shard_keys)
            if info.write_to:
                w_slots = list(dict.fromkeys([*w_slots, *info.write_to]))
            picked = None
        return {
            "collection": collection,
            "feature": str(feature),
            "known": True,
            "fanout": False,
            "canonical": canonical,
            "state": info.state.value,
            "shard_keys": list(info.shard_keys),
            "write_to": None if info.write_to is None else list(info.write_to),
            "aliases": list(info.aliases),
            "read_slots": r_slots,
            "write_slots": w_slots,
            "picked_slot": picked,
        }

    def prometheus_metrics(self, live: bool = False) -> str:
        items = self._all_collection_stats(live=live)
        routing = self._metrics.snapshot() if self._metrics is not None else None
        return render_prometheus(items, routing)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._jobs.list_jobs()

    def create_collection(
        self,
        collection_name: str,
        feature_field: str,
        feature_type: str = "keyword",
        n_slots: int = 48,
        vector_size: int = 128,
        distance: str = "Cosine",
        feature_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")
        if not feature_field.strip():
            raise ValueError("feature_field must not be blank")
        if vector_size < 1:
            raise ValueError("vector_size must be at least 1")
        dist = _parse_distance(distance)
        doc = services.create_sharded_collection(
            self._store(collection_name),
            vectors,
            collection_name,
            feature_field,
            feature_type,
            n_slots=n_slots,
            feature_counts=feature_counts,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )
        return {"map": _map_dict(doc)}

    def refresh_stats(self, collection: str) -> dict[str, Any]:
        self._require_writable()
        doc = services.refresh_stats(self._store(collection), self._require_vectors())
        return {
            "stats": collection_stats(doc, now=self._now()).to_dict(),
            "map": _map_dict(doc),
        }

    def add_slot(self, collection: str, slot: str) -> dict[str, Any]:
        self._require_writable()
        doc = services.add_slot(self._store(collection), self._require_vectors(), slot)
        return {"map": _map_dict(doc)}

    def add_feature(self, collection: str, value: int | str) -> dict[str, Any]:
        self._require_writable()
        info = services.add_feature(self._store(collection), value)
        return {"feature": str(value), "shard_keys": list(info.shard_keys)}

    def migrate(
        self,
        collection: str,
        feature: int | str,
        to_slot: str,
        grace_seconds: int = 60,
        background: bool = True,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        store = self._store(collection)
        services.check_grace_seconds(grace_seconds)

        def task() -> dict[str, Any]:
            services.migrate(
                store, vectors, feature, [to_slot], self._now, self._sleep, grace_seconds
            )
            return {"map": _map_dict(store.load())}

        if background:
            return self._jobs.submit(collection, "migrate", str(feature), task)
        return task()

    def split(
        self,
        collection: str,
        feature: int | str,
        partitions: int,
        grace_seconds: int = 60,
        background: bool = True,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        store = self._store(collection)
        services.check_grace_seconds(grace_seconds)
        targets = services.split_targets(store.load(), feature, partitions)

        def task() -> dict[str, Any]:
            services.migrate(
                store, vectors, feature, targets, self._now, self._sleep, grace_seconds
            )
            return {"targets": targets, "map": _map_dict(store.load())}

        if background:
            return self._jobs.submit(collection, "split", str(feature), task)
        return task()

    def apply_rebalance(
        self,
        collection: str,
        grace_seconds: int = 60,
        background: bool = True,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        store = self._store(collection)
        services.check_grace_seconds(grace_seconds)
        plan = services.plan_rebalance(store, vectors)

        def task() -> dict[str, Any]:
            services.apply_rebalance(
                store, vectors, plan, self._now, self._sleep, grace_seconds
            )
            return {
                "moves": len(plan.moves),
                "reason": plan.reason.value,
                "map": _map_dict(store.load()),
            }

        if background:
            return self._jobs.submit(collection, "rebalance", None, task)
        return task()

    def start_move(
        self,
        collection: str,
        feature: int | str,
        targets: list[str],
        grace_seconds: int = 60,
    ) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        services.start_move(store, feature, targets, self._now(), grace_seconds)
        return {"map": _map_dict(store.load())}

    def copy_feature(
        self, collection: str, feature: int | str, batch: int = 256
    ) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        copied = services.copy_feature(
            store, self._require_vectors(), feature, self._now(), batch=batch
        )
        return {"copied": copied, "map": _map_dict(store.load())}

    def open_dual(
        self, collection: str, feature: int | str, grace_seconds: int = 60
    ) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        services.open_dual(store, feature, self._now(), grace_seconds)
        return {"map": _map_dict(store.load())}

    def cutover(
        self, collection: str, feature: int | str, grace_seconds: int = 60
    ) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        result = services.cutover(
            store, self._require_vectors(), feature, self._now(), grace_seconds
        )
        return {
            "old_slots": list(result.old_slots),
            "copied": result.copied,
            "removed": result.removed,
            "map": _map_dict(store.load()),
        }

    def finish_move(self, collection: str, feature: int | str) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        services.finish_move(store, self._require_vectors(), feature, self._now())
        return {"map": _map_dict(store.load())}

    def resume_move(
        self,
        collection: str,
        feature: int | str,
        grace_seconds: int = 60,
        background: bool = True,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        store = self._store(collection)
        services.check_grace_seconds(grace_seconds)

        def task() -> dict[str, Any]:
            services.resume_move(
                store, vectors, feature, self._now, self._sleep, grace_seconds
            )
            return {"map": _map_dict(store.load())}

        if background:
            return self._jobs.submit(collection, "resume_move", str(feature), task)
        return task()

    def alias_feature(
        self,
        collection: str,
        old: int | str,
        new: int | str,
        grace_seconds: int = 60,
        background: bool = True,
    ) -> dict[str, Any]:
        self._require_writable()
        vectors = self._require_vectors()
        store = self._store(collection)
        services.check_grace_seconds(grace_seconds)

        def task() -> dict[str, Any]:
            rewritten = services.alias_feature(
                store, vectors, old, new, self._sleep, grace_seconds
            )
            return {"rewritten": rewritten, "map": _map_dict(store.load())}

        if background:
            return self._jobs.submit(collection, "alias_feature", str(old), task)
        return task()

    def drop_alias(self, collection: str, old: int | str) -> dict[str, Any]:
        self._require_writable()
        store = self._store(collection)
        rewritten = services.drop_alias(store, self._require_vectors(), old)
        return {
            "dropped": str(old),
            "rewritten": rewritten,
            "map": _map_dict(store.load()),
        }

    def _discover_collections(self) -> list[str]:
        return list_map_collections(self._folder)

    def _all_collection_stats(self, live: bool = False):
        stats_list = []
        for name in self._discover_collections():
            try:
                doc = self._store(name).load()
            except MapNotFound:
                continue
            snapshot = (
                services.collect_load(self._require_vectors(), doc)
                if live and self._vectors is not None
                else None
            )
            stats_list.append(collection_stats(doc, snapshot=snapshot, now=self._now()))
        return stats_list

    def _store(self, collection: str):
        return self._folder.store_for(collection)

    def _require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("dashboard is running in read-only mode")

    def _require_vectors(self) -> VectorStore:
        if self._vectors is None:
            raise MapUnavailable("no Qdrant vector store configured")
        return self._vectors


class DashboardServer:
    """HTTP server exposing metrics, JSON API, and the management dashboard UI."""

    def __init__(
        self,
        httpd: ThreadingHTTPServer,
        controller: DashboardController,
    ) -> None:
        self._httpd = httpd
        self.controller = controller
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def create_dashboard_server(
    source: DashboardController | ShardedQdrantClient,
    host: str = "127.0.0.1",
    port: int = 8080,
    read_only: bool = False,
    api_token: str | None = None,
) -> DashboardServer:
    """Create a ThreadingHTTPServer bound to host:port for metrics and dashboard UI."""
    if isinstance(source, DashboardController):
        if read_only and not source.read_only:
            controller = DashboardController(
                folder=source._folder,
                vectors=source._vectors,
                metrics=source._metrics,
                now_fn=source._now,
                sleep_fn=source._sleep,
                read_only=True,
            )
            controller._jobs = source._jobs
        else:
            controller = source
    else:
        controller = DashboardController.from_client(source, read_only=read_only)
    handler_cls = _make_handler(controller, api_token)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    return DashboardServer(httpd, controller)


def serve_dashboard(
    source: DashboardController | ShardedQdrantClient,
    host: str = "127.0.0.1",
    port: int = 8080,
    read_only: bool = False,
    api_token: str | None = None,
) -> None:
    """Run the metrics and management dashboard HTTP server until interrupted."""
    server = create_dashboard_server(
        source, host=host, port=port, read_only=read_only, api_token=api_token
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def _make_handler(
    controller: DashboardController, api_token: str | None
) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            if method == "GET" and path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    DASHBOARD_HTML.encode("utf-8"),
                )
                return

            if method == "GET" and path == "/favicon.ico":
                self.send_response(int(HTTPStatus.NO_CONTENT))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if method == "GET" and path == "/healthz":
                self._handle_json(lambda: (HTTPStatus.OK, controller.health()))
                return

            if not self._authorized():
                self._send_json(
                    HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
                )
                return

            if method == "GET" and path == "/metrics":
                live = _bool_param(query, "live")
                self._handle_text(
                    "text/plain; version=0.0.4; charset=utf-8",
                    lambda: controller.prometheus_metrics(live=live),
                )
                return

            self._handle_json(lambda: self._route_api(method, path, query))

        def _route_api(
            self, method: str, path: str, query: dict[str, list[str]]
        ) -> tuple[HTTPStatus, dict[str, Any]]:
            parts = [unquote(part) for part in path.strip("/").split("/") if part]
            if not parts or parts[0] != "api":
                raise _NotFoundError(f"unknown route {path}")

            if method == "GET" and parts == ["api", "overview"]:
                return HTTPStatus.OK, controller.overview(live=_bool_param(query, "live"))

            if method == "GET" and parts == ["api", "jobs"]:
                return HTTPStatus.OK, {"jobs": controller.list_jobs()}

            if parts == ["api", "collections"]:
                if method == "GET":
                    return HTTPStatus.OK, controller.overview(
                        live=_bool_param(query, "live")
                    )
                if method == "POST":
                    body = self._read_json()
                    created = controller.create_collection(
                        collection_name=str(body.get("collection_name", "")),
                        feature_field=str(body.get("feature_field", "")),
                        feature_type=str(body.get("feature_type", "keyword")),
                        n_slots=_int_field(body, "n_slots", 48),
                        vector_size=_int_field(body, "vector_size", 128),
                        distance=str(body.get("distance", "Cosine")),
                        feature_counts=body.get("feature_counts"),
                    )
                    return HTTPStatus.CREATED, created

            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "collections":
                collection = parts[2]
                tail = parts[3:]
                return self._route_collection(method, collection, tail, query)

            raise _NotFoundError(f"unknown route {path}")

        def _route_collection(
            self,
            method: str,
            collection: str,
            tail: list[str],
            query: dict[str, list[str]],
        ) -> tuple[HTTPStatus, dict[str, Any]]:
            if not tail and method == "GET":
                return HTTPStatus.OK, controller.collection_detail(
                    collection, live=_bool_param(query, "live")
                )
            if tail == ["map"] and method == "GET":
                return HTTPStatus.OK, controller.get_map_dict(collection)
            if tail == ["plan"] and method == "GET":
                return HTTPStatus.OK, controller.preview_rebalance(collection)
            if tail == ["route"] and method == "GET":
                raw_feature = _first_param(query, "feature")
                if raw_feature is None:
                    raise ValueError("feature query parameter is required")
                point_id = _first_param(query, "point_id")
                return HTTPStatus.OK, controller.inspect_route(
                    collection, raw_feature, point_id=point_id
                )

            if method == "POST":
                body = self._read_json()
                if tail == ["refresh"]:
                    return HTTPStatus.OK, controller.refresh_stats(collection)
                if tail == ["slots"]:
                    return HTTPStatus.OK, controller.add_slot(
                        collection, str(body.get("slot", ""))
                    )
                if tail == ["features"]:
                    if "value" not in body:
                        raise ValueError("value is required")
                    return HTTPStatus.OK, controller.add_feature(
                        collection, body["value"]
                    )
                if tail == ["migrate"]:
                    if "feature" not in body or "to_slot" not in body:
                        raise ValueError("feature and to_slot are required")
                    return HTTPStatus.OK, controller.migrate(
                        collection,
                        body["feature"],
                        str(body["to_slot"]),
                        grace_seconds=_int_field(body, "grace_seconds", 60),
                        background=bool(body.get("background", True)),
                    )
                if tail == ["split"]:
                    if "feature" not in body or "partitions" not in body:
                        raise ValueError("feature and partitions are required")
                    return HTTPStatus.OK, controller.split(
                        collection,
                        body["feature"],
                        _int_field(body, "partitions", 2),
                        grace_seconds=_int_field(body, "grace_seconds", 60),
                        background=bool(body.get("background", True)),
                    )
                if tail == ["rebalance"]:
                    return HTTPStatus.OK, controller.apply_rebalance(
                        collection,
                        grace_seconds=_int_field(body, "grace_seconds", 60),
                        background=bool(body.get("background", True)),
                    )
                if tail == ["aliases"]:
                    if "old" not in body or "new" not in body:
                        raise ValueError("old and new are required")
                    return HTTPStatus.OK, controller.alias_feature(
                        collection,
                        body["old"],
                        body["new"],
                        grace_seconds=_int_field(body, "grace_seconds", 60),
                        background=bool(body.get("background", True)),
                    )
                if len(tail) == 3 and tail[0] == "moves":
                    feature = tail[1]
                    step = tail[2]
                    if step == "start":
                        targets = body.get("targets")
                        if not isinstance(targets, list):
                            raise ValueError("targets must be a list of slots")
                        return HTTPStatus.OK, controller.start_move(
                            collection,
                            feature,
                            [str(item) for item in targets],
                            grace_seconds=_int_field(body, "grace_seconds", 60),
                        )
                    if step == "copy":
                        return HTTPStatus.OK, controller.copy_feature(
                            collection,
                            feature,
                            batch=_int_field(body, "batch", 256),
                        )
                    if step == "open-dual":
                        return HTTPStatus.OK, controller.open_dual(
                            collection,
                            feature,
                            grace_seconds=_int_field(body, "grace_seconds", 60),
                        )
                    if step == "cutover":
                        return HTTPStatus.OK, controller.cutover(
                            collection,
                            feature,
                            grace_seconds=_int_field(body, "grace_seconds", 60),
                        )
                    if step == "finish":
                        return HTTPStatus.OK, controller.finish_move(
                            collection, feature
                        )
                    if step == "resume":
                        return HTTPStatus.OK, controller.resume_move(
                            collection,
                            feature,
                            grace_seconds=_int_field(body, "grace_seconds", 60),
                            background=bool(body.get("background", True)),
                        )

            if method == "DELETE" and len(tail) == 2 and tail[0] == "aliases":
                return HTTPStatus.OK, controller.drop_alias(collection, tail[1])

            raise _NotFoundError(f"unknown route for collection {collection}")

        def _authorized(self) -> bool:
            if not api_token:
                return True
            header = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not header.startswith(prefix):
                return False
            return hmac.compare_digest(header[len(prefix) :], api_token)

        def _read_json(self) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length")
            if not raw_len:
                return {}
            length = int(raw_len)
            if length > _MAX_BODY_BYTES:
                raise ValueError("request body exceeds 1 MiB")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("request body must be a JSON object")
            return parsed

        def _handle_text(self, content_type: str, action: Callable[[], str]) -> None:
            try:
                text = action()
            except Exception as exc:
                status, payload = _error_response(exc)
                self._send_json(status, payload)
                return
            self._send_bytes(HTTPStatus.OK, content_type, text.encode("utf-8"))

        def _handle_json(
            self, action: Callable[[], tuple[HTTPStatus, dict[str, Any]]]
        ) -> None:
            try:
                status, payload = action()
            except Exception as exc:
                status, payload = _error_response(exc)
            self._send_json(status, payload)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, indent=2).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", encoded)

        def _send_bytes(
            self, status: HTTPStatus, content_type: str, body: bytes
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


class _NotFoundError(Exception):
    pass


def _error_response(exc: Exception) -> tuple[HTTPStatus, dict[str, Any]]:
    if isinstance(exc, _NotFoundError):
        return HTTPStatus.NOT_FOUND, {"error": str(exc)}
    if isinstance(exc, PermissionError):
        return HTTPStatus.FORBIDDEN, {"error": str(exc)}
    if isinstance(exc, MapNotFound):
        return HTTPStatus.NOT_FOUND, {"error": str(exc)}
    if isinstance(exc, (MoveNotReady, VersionConflict)):
        return HTTPStatus.CONFLICT, {"error": str(exc)}
    if isinstance(exc, MapUnavailable):
        return HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)}
    if isinstance(exc, (InvalidMove, ShardingError, ValueError, json.JSONDecodeError)):
        return HTTPStatus.BAD_REQUEST, {"error": str(exc)}
    if isinstance(exc, UnexpectedResponse):
        status = _http_status_for_code(exc.status_code)
        return status, {"error": str(exc)}
    if isinstance(exc, grpc.RpcError):
        return _http_status_for_grpc(exc.code()), {"error": exc.details() or str(exc)}
    return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}


def _http_status_for_code(code: int | None) -> HTTPStatus:
    if code is not None and 400 <= code < 600:
        try:
            return HTTPStatus(code)
        except ValueError:
            return HTTPStatus.BAD_GATEWAY
    return HTTPStatus.BAD_GATEWAY


def _http_status_for_grpc(code: grpc.StatusCode | None) -> HTTPStatus:
    mapping = {
        grpc.StatusCode.INVALID_ARGUMENT: HTTPStatus.BAD_REQUEST,
        grpc.StatusCode.ALREADY_EXISTS: HTTPStatus.CONFLICT,
        grpc.StatusCode.NOT_FOUND: HTTPStatus.NOT_FOUND,
        grpc.StatusCode.PERMISSION_DENIED: HTTPStatus.FORBIDDEN,
        grpc.StatusCode.UNAUTHENTICATED: HTTPStatus.UNAUTHORIZED,
        grpc.StatusCode.UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED: HTTPStatus.GATEWAY_TIMEOUT,
    }
    return mapping.get(code, HTTPStatus.BAD_GATEWAY)


def _map_dict(doc: MapDoc) -> dict[str, Any]:
    return json.loads(map_to_json(doc))


def _bool_param(query: dict[str, list[str]], key: str) -> bool:
    raw = _first_param(query, key)
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes"}


def _first_param(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _int_field(body: dict[str, Any], key: str, default: int) -> int:
    value = body.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _parse_distance(raw: str) -> Distance:
    key = raw.strip().lower()
    table = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }
    dist = table.get(key)
    if dist is None:
        raise ValueError(f"unknown distance {raw!r}; use Cosine, Dot, Euclid, or Manhattan")
    return dist


def _simulate_plan_loads(doc: MapDoc, snapshot, plan) -> dict[str, int]:
    totals = {slot: sum(snapshot.counts.get(slot, {}).values()) for slot in doc.slots}
    for order in plan.moves:
        found = doc.find(order.feature)
        if found is None:
            continue
        name, info = found
        names = {name, *(str(alias) for alias in info.aliases)}
        for slot in info.shard_keys:
            facets = snapshot.counts.get(slot, {})
            removed = sum(count for key, count in facets.items() if str(key) in names)
            if slot in totals:
                totals[slot] = max(0, totals[slot] - removed)
        base, extra = divmod(order.points, len(order.to_slots))
        for index, slot in enumerate(order.to_slots):
            totals[slot] = totals.get(slot, 0) + base + (1 if index < extra else 0)
    return totals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the Qdrant Custom Sharding metrics and management dashboard."
    )
    parser.add_argument(
        "--map-uri",
        default=os.environ.get("MAP_URI"),
        help="Map folder URI (local path, file://, or s3://bucket/prefix/). Defaults to MAP_URI.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL"),
        help="Qdrant URL (e.g. http://localhost:6333). Omit for map-only inspection mode.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.environ.get("QDRANT_API_KEY"),
        help="Optional Qdrant API key.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Disable mutating management endpoints.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("DASHBOARD_API_TOKEN"),
        help="Optional Bearer token required for /api/* and /metrics.",
    )
    args = parser.parse_args(argv)
    if not args.map_uri:
        parser.error("--map-uri or MAP_URI is required")

    if args.qdrant_url:
        client = ShardedQdrantClient(
            url=args.qdrant_url,
            api_key=args.qdrant_api_key,
            map_uri=args.map_uri,
        )
        source: DashboardController | ShardedQdrantClient = client
    else:
        source = DashboardController(
            folder=open_map_folder(args.map_uri),
            vectors=None,
            read_only=args.read_only,
        )

    server = create_dashboard_server(
        source,
        host=args.host,
        port=args.port,
        read_only=args.read_only,
        api_token=args.api_token,
    )
    print(f"Qdrant Custom Sharding dashboard listening on {server.url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
