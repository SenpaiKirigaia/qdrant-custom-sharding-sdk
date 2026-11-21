import json
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from qdrant_custom_sharding.dashboard import (
    DashboardController,
    create_dashboard_server,
)
from qdrant_custom_sharding.metrics import InMemoryMetrics
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, Point
from qdrant_custom_sharding.services import add_feature, create_sharded_collection
from tests.fakes import FakeMapFolder, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _setup_controller(read_only: bool = False):
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    metrics = InMemoryMetrics()
    clock = [NOW]

    def now_fn() -> datetime:
        return clock[0]

    def sleep_fn(seconds: int) -> None:
        clock[0] = clock[0] + timedelta(seconds=seconds)

    create_sharded_collection(
        folder.store_for("catalog"),
        vectors,
        "catalog",
        "category",
        "keyword",
        n_slots=4,
        vectors_config={"size": 4, "distance": "Cosine"},
    )
    add_feature(folder.store_for("catalog"), "books")
    add_feature(folder.store_for("catalog"), "toys")
    for point_id in range(1, 21):
        vectors.upsert(
            "catalog",
            "s0",
            [Point(id=point_id, vector=[0.1, 0.2, 0.3, 0.4], payload={"category": "books"})],
        )
    for point_id in range(21, 25):
        vectors.upsert(
            "catalog",
            "s1",
            [Point(id=point_id, vector=[0.1, 0.2, 0.3, 0.4], payload={"category": "toys"})],
        )
    metrics.record_read("catalog", "query_points", fanout=False, slot_count=1)
    metrics.record_read("catalog", "count", fanout=True, slot_count=4)
    controller = DashboardController(
        folder=folder,
        vectors=vectors,
        metrics=metrics,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        read_only=read_only,
    )
    return controller, folder, vectors, clock


def test_controller_overview_detail_plan_and_route_inspector():
    controller, _folder, _vectors, _clock = _setup_controller()
    controller.refresh_stats("catalog")

    overview = controller.overview()
    assert overview["collection_count"] == 1
    assert overview["collections"][0]["collection"] == "catalog"
    assert overview["collections"][0]["total_points"] == 24
    assert overview["routing"]["total_routed_reads"] == 1
    assert overview["routing"]["total_fanout_reads"] == 1

    detail = controller.collection_detail("catalog", live=True)
    assert detail["stats"]["source"] == "live"
    assert detail["stats"]["total_points"] == 24
    assert detail["stats"]["imbalance_ratio"] > 1.5
    assert detail["map"]["collection"] == "catalog"

    plan = controller.preview_rebalance("catalog")
    assert plan["collection"] == "catalog"
    assert isinstance(plan["moves"], list)
    assert len(plan["split_recommendations"]) == 1
    assert plan["split_recommendations"][0]["feature"] == "books"

    route = controller.inspect_route("catalog", "books", point_id=7)
    assert route["known"] is True
    assert route["canonical"] == "books"
    assert route["read_slots"] == ["s0"]
    assert route["write_slots"] == ["s0"]
    assert route["picked_slot"] == "s0"

    missing_route = controller.inspect_route("catalog", "unknown")
    assert missing_route["known"] is False
    assert missing_route["fanout"] is True


def test_controller_management_operations_and_background_jobs():
    controller, folder, _vectors, _clock = _setup_controller()
    controller.add_slot("catalog", "s4")
    assert "s4" in folder.store_for("catalog").load().slots

    added = controller.add_feature("catalog", "music")
    assert added["feature"] == "music"
    assert len(added["shard_keys"]) == 1

    # Synchronous migrate
    migrated = controller.migrate("catalog", "toys", "s2", grace_seconds=0, background=False)
    assert migrated["map"]["features"]["toys"]["shard_keys"] == ["s2"]

    # Background split job
    job = controller.split("catalog", "books", partitions=2, grace_seconds=0, background=True)
    assert job["action"] == "split"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        jobs = controller.list_jobs()
        current = next(item for item in jobs if item["id"] == job["id"])
        if current["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert current["status"] == "completed"
    assert len(folder.store_for("catalog").load().features["books"].shard_keys) == 2

    # Alias and drop alias
    aliased = controller.alias_feature("catalog", "toys", "games", grace_seconds=0, background=False)
    assert aliased["rewritten"] == 4
    dropped = controller.drop_alias("catalog", "toys")
    assert dropped["dropped"] == "toys"


def test_controller_step_by_step_move_and_map_only_mode():
    controller, folder, _vectors, _clock = _setup_controller()
    controller.start_move("catalog", "toys", ["s3"], grace_seconds=0)
    assert folder.store_for("catalog").load().features["toys"].state is FeatureState.copying
    copied = controller.copy_feature("catalog", "toys")
    assert copied["copied"] == 4
    controller.open_dual("catalog", "toys", grace_seconds=0)
    cut = controller.cutover("catalog", "toys", grace_seconds=0)
    assert cut["old_slots"] == ["s1"]
    controller.finish_move("catalog", "toys")
    assert folder.store_for("catalog").load().features["toys"].shard_keys == ["s3"]

    # Map-only controller works for stats/overview/prometheus without Qdrant.
    map_only = DashboardController(folder=folder, vectors=None)
    assert map_only.overview()["collection_count"] == 1
    assert "qdrant_sharding_slot_points" in map_only.prometheus_metrics()


def test_http_server_serves_ui_metrics_api_auth_and_errors():
    controller, _folder, _vectors, _clock = _setup_controller()
    controller.refresh_stats("catalog")
    server = create_dashboard_server(controller, host="127.0.0.1", port=0)
    server.start_in_thread()
    base = server.url
    try:
        with httpx.Client(base_url=base, timeout=5.0) as http:
            home = http.get("/")
            assert home.status_code == 200
            assert "text/html" in home.headers["content-type"]
            assert "Qdrant Custom Sharding" in home.text

            health = http.get("/healthz")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            prom = http.get("/metrics")
            assert prom.status_code == 200
            assert "qdrant_sharding_collection_points_total" in prom.text
            assert 'collection="catalog"' in prom.text

            overview = http.get("/api/overview")
            assert overview.status_code == 200
            assert overview.json()["collection_count"] == 1

            detail = http.get("/api/collections/catalog?live=1")
            assert detail.status_code == 200
            assert detail.json()["stats"]["total_points"] == 24

            missing = http.get("/api/collections/does_not_exist")
            assert missing.status_code == 404

            plan = http.get("/api/collections/catalog/plan")
            assert plan.status_code == 200
            assert plan.json()["collection"] == "catalog"

            route = http.get("/api/collections/catalog/route?feature=books&point_id=1")
            assert route.status_code == 200
            assert route.json()["picked_slot"] == "s0"

            slot_resp = http.post("/api/collections/catalog/slots", json={"slot": "s9"})
            assert slot_resp.status_code == 200
            assert "s9" in slot_resp.json()["map"]["slots"]

            bad_slot = http.post("/api/collections/catalog/slots", json={"slot": "  "})
            assert bad_slot.status_code == 400

            feat_resp = http.post("/api/collections/catalog/features", json={"value": "art"})
            assert feat_resp.status_code == 200
            assert feat_resp.json()["feature"] == "art"

            create_resp = http.post(
                "/api/collections",
                json={
                    "collection_name": "people",
                    "feature_field": "subject_id",
                    "feature_type": "integer",
                    "n_slots": 3,
                    "vector_size": 4,
                    "distance": "Cosine",
                },
            )
            assert create_resp.status_code == 201
            assert create_resp.json()["map"]["collection"] == "people"

            # Start move with grace=60 then try early copy -> 409 MoveNotReady
            start_resp = http.post(
                "/api/collections/catalog/moves/toys/start",
                json={"targets": ["s3"], "grace_seconds": 60},
            )
            assert start_resp.status_code == 200
            early_copy = http.post("/api/collections/catalog/moves/toys/copy", json={})
            assert early_copy.status_code == 409

            jobs_resp = http.get("/api/jobs")
            assert jobs_resp.status_code == 200
            assert isinstance(jobs_resp.json()["jobs"], list)
    finally:
        server.shutdown()


def test_http_server_enforces_read_only_and_bearer_token():
    controller, _folder, _vectors, _clock = _setup_controller(read_only=True)
    server = create_dashboard_server(
        controller, host="127.0.0.1", port=0, api_token="secret-token"
    )
    server.start_in_thread()
    try:
        with httpx.Client(base_url=server.url, timeout=5.0) as http:
            assert http.get("/healthz").status_code == 200
            assert http.get("/api/overview").status_code == 401
            assert http.get("/metrics").status_code == 401

            headers = {"Authorization": "Bearer secret-token"}
            assert http.get("/api/overview", headers=headers).status_code == 200
            assert http.get("/metrics", headers=headers).status_code == 200

            denied = http.post(
                "/api/collections/catalog/slots",
                json={"slot": "s8"},
                headers=headers,
            )
            assert denied.status_code == 403
    finally:
        server.shutdown()
