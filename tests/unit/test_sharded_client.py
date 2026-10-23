import asyncio
import linecache
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue, ShardingMethod

from qdrant_custom_sharding.async_sharded_client import AsyncShardedQdrantClient
from qdrant_custom_sharding.errors import WriteDenied
from qdrant_custom_sharding.models import FeatureInfo, MapDoc
from qdrant_custom_sharding.calls import READ_FANOUT, STALE_USED, bind_call, supported_kwargs
from qdrant_custom_sharding.sharded_client import ShardedQdrantClient
from qdrant_custom_sharding.sharding_warnings import FanOutWarning, ShardingWarning, StaleMapWarning
from tests.fakes import FakeMapFolder
from tests.local_qdrant import compatibility_kwargs


def test_bind_call_mixes_positional_keyword_and_extra():
    points = ["point"]
    flat = bind_call(
        QdrantClient.upsert,
        object(),
        ("items", points),
        {"wait": False, "made_up": 1},
    )
    assert flat["collection_name"] == "items"
    assert flat["points"] is points
    assert flat["wait"] is False
    assert flat["shard_key_selector"] is None
    assert flat["ordering"] is None
    assert flat["made_up"] == 1
    assert "self" not in flat
    assert "kwargs" not in flat


def test_bind_call_accepts_keywords():
    flat = bind_call(
        QdrantClient.count,
        object(),
        (),
        {"collection_name": "items", "exact": False},
    )
    assert flat["collection_name"] == "items"
    assert flat["count_filter"] is None
    assert flat["exact"] is False
    assert flat["shard_key_selector"] is None
    assert "kwargs" not in flat


def _stored_map() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
    )


def _with_timeout(timeout=None):
    return timeout


def _without_timeout(**kwargs):
    return kwargs


def test_supported_kwargs_keeps_only_declared_names():
    given = {"timeout": 5, "update_filter": None}
    assert supported_kwargs(_with_timeout, given) == {"timeout": 5}
    assert supported_kwargs(_without_timeout, given) == {}


def _client(folder: FakeMapFolder, now):
    return ShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=now,
        map_ttl_seconds=30,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )


def test_map_for_passes_through_when_the_caller_or_the_folder_opts_out():
    folder = FakeMapFolder()
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    client = _client(folder, lambda: now)
    assert client._router.map_for("items", "s0", write=True) is None
    assert client._router.map_for("items", None, write=False) is None
    folder.store_for("items").save(_stored_map())
    client._router._folder = None
    assert client._router.map_for("items", None, write=True) is None


def _through_map(client: ShardedQdrantClient):
    return client._router.map_for("items", None, write=True)


def _as_override(client: ShardedQdrantClient):
    return _through_map(client)


def test_map_for_returns_a_fresh_map_and_warns_on_a_stale_one():
    folder = FakeMapFolder()
    clock = {"now": datetime(2020, 1, 1, tzinfo=timezone.utc)}
    client = _client(folder, lambda: clock["now"])
    folder.store_for("items").save(_stored_map())
    fresh = client._router.map_for("items", None, write=False)
    assert fresh is not None
    assert fresh.feature_field == "feature"
    folder.store_for("items").fail_with_unavailable = True
    clock["now"] = clock["now"] + timedelta(seconds=31)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stale = _as_override(client)
    assert stale is not None
    assert stale.feature_field == "feature"
    assert len(caught) == 1
    assert caught[0].category is StaleMapWarning
    text = linecache.getline(caught[0].filename, caught[0].lineno)
    assert "_as_override(client)" in text


def test_map_for_fans_out_custom_reads_and_denies_custom_writes():
    folder = FakeMapFolder()
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    client = _client(folder, lambda: now)
    folder.store_for("items").fail_with_unavailable = True
    client._router._methods["items"] = "auto"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert client._router.map_for("items", None, write=False) is None
        assert client._router.map_for("items", None, write=True) is None
    assert caught == []
    client._router._methods["items"] = "custom"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert client._router.map_for("items", None, write=False) is None
    assert any(issubclass(item.category, StaleMapWarning) for item in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StaleMapWarning)
        with pytest.raises(WriteDenied, match="unavailable"):
            client._router.map_for("items", None, write=True)


def _slots_map() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0, "s1": 0},
    )


class _Point:
    def __init__(self, point_id, payload, shard_key=None):
        self.id = point_id
        self.payload = payload
        self.shard_key = shard_key


def _collection_info():
    params = SimpleNamespace(sharding_method=ShardingMethod.CUSTOM)
    return SimpleNamespace(config=SimpleNamespace(params=params))


def _toy_filter() -> Filter:
    return Filter(must=FieldCondition(key="feature", match=MatchValue(value="toys")))


def _books_filter() -> Filter:
    return Filter(must=FieldCondition(key="feature", match=MatchValue(value="books")))


def _async_client(folder: FakeMapFolder, now):
    return AsyncShardedQdrantClient(
        url="http://127.0.0.1:1",
        map_folder=folder,
        now_fn=now,
        map_ttl_seconds=30,
        prefer_grpc=False,
        timeout=1,
        **compatibility_kwargs(),
    )


def _ready(async_client=False):
    folder = FakeMapFolder()
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    if async_client:
        client = _async_client(folder, lambda: now)
    else:
        client = _client(folder, lambda: now)
    folder.store_for("items").save(_slots_map())
    return client, folder


def _stub(client, calls, records):
    def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        if method.__name__ == "retrieve":
            return records
        if method.__name__ == "get_collection":
            return _collection_info()
        return "ok"

    client._router._invoke = invoke


def _stub_async(client, calls, records):
    async def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        if method.__name__ == "retrieve":
            return records
        if method.__name__ == "get_collection":
            return _collection_info()
        return "ok"

    client._router._invoke = invoke


def _stale(caught) -> bool:
    return any(item.category is StaleMapWarning and str(item.message) == STALE_USED for item in caught)


def test_id_write_uses_cached_slots_when_the_reload_fails():
    client, folder = _ready()
    client._router.map_for("items", None, write=True)
    folder.store_for("items").fail_with_unavailable = True
    calls = []
    _stub(client, calls, [_Point(7, {"feature": "toys"}, shard_key="s1")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.set_payload("items", {"tag": "x"}, [7])
    assert [name for name, _flat in calls] == ["retrieve", "set_payload"]
    assert calls[1][1]["shard_key_selector"] == ["s1"]
    assert _stale(caught)


def test_filter_read_keeps_the_cached_map_when_the_reload_fails():
    client, folder = _ready()
    client._router.map_for("items", None, write=False)
    folder.store_for("items").fail_with_unavailable = True
    calls = []
    _stub(client, calls, [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.count("items", count_filter=_toy_filter())
    assert [name for name, _flat in calls] == ["count"]
    assert calls[0][1]["shard_key_selector"] is None
    assert _stale(caught)
    assert any(item.category is FanOutWarning and str(item.message) == READ_FANOUT for item in caught)


def test_id_write_fans_out_when_the_point_has_no_feature_field():
    client, _folder = _ready()
    calls = []
    _stub(client, calls, [_Point(7, {"note": "kept"}, shard_key="s0")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FanOutWarning)
        client.set_payload("items", {"tag": "x"}, [7])
    writes = [flat for name, flat in calls if name == "set_payload"]
    assert len(writes) == 1
    assert writes[0]["shard_key_selector"] == ["s0"]
    assert writes[0]["points"] == [7]


def test_async_id_write_uses_cached_slots_when_the_reload_fails():
    asyncio.run(_async_id_write_reload())


async def _async_id_write_reload():
    client, folder = _ready(async_client=True)
    await client._router.map_for("items", None, write=True)
    folder.store_for("items").fail_with_unavailable = True
    calls = []
    _stub_async(client, calls, [_Point(7, {"feature": "toys"}, shard_key="s1")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.set_payload("items", {"tag": "x"}, [7])
    assert [name for name, _flat in calls] == ["retrieve", "set_payload"]
    assert calls[1][1]["shard_key_selector"] == ["s1"]
    assert _stale(caught)


def test_async_filter_read_keeps_the_cached_map_when_the_reload_fails():
    asyncio.run(_async_filter_read_reload())


async def _async_filter_read_reload():
    client, folder = _ready(async_client=True)
    await client._router.map_for("items", None, write=False)
    folder.store_for("items").fail_with_unavailable = True
    calls = []
    _stub_async(client, calls, [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.count("items", count_filter=_toy_filter())
    assert [name for name, _flat in calls] == ["count"]
    assert calls[0][1]["shard_key_selector"] is None
    assert _stale(caught)
    assert any(item.category is FanOutWarning and str(item.message) == READ_FANOUT for item in caught)


def test_id_write_probes_each_slot_when_retrieve_has_no_shard_key():
    client, _folder = _ready()
    calls = []

    def invoke(method, flat):
        calls.append((method.__name__, dict(flat)))
        if method.__name__ == "retrieve":
            return [_Point(7, {"note": "kept"})]
        if flat.get("shard_key_selector") == ["s0"]:
            raise UnexpectedResponse(404, "Not Found", b"no point", httpx.Headers())
        return "ok"

    client._router._invoke = invoke
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FanOutWarning)
        assert client.set_payload("items", {"tag": "x"}, [7]) == "ok"
    writes = [flat["shard_key_selector"] for name, flat in calls if name == "set_payload"]
    assert writes == [["s0"], ["s1"]]


def test_async_id_write_fans_out_when_the_point_has_no_feature_field():
    asyncio.run(_async_missing_feature())


async def _async_missing_feature():
    client, _folder = _ready(async_client=True)
    calls = []
    _stub_async(client, calls, [_Point(7, {"note": "kept"}, shard_key="s0")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FanOutWarning)
        await client.set_payload("items", {"tag": "x"}, [7])
    writes = [flat for name, flat in calls if name == "set_payload"]
    assert len(writes) == 1
    assert writes[0]["shard_key_selector"] == ["s0"]
    assert writes[0]["points"] == [7]


def _books_selector(client, method_name: str, call) -> None:
    calls = []
    _stub(client, calls, [])
    call()
    matched = [flat for name, flat in calls if name == method_name]
    assert len(matched) == 1
    assert matched[0]["shard_key_selector"] == ["s0"]


def test_query_points_groups_selects_the_feature_slot():
    client, _folder = _ready()
    _books_selector(
        client,
        "query_points_groups",
        lambda: client.query_points_groups(
            "items",
            "feature",
            query=[0.1, 0.2, 0.3, 0.4],
            query_filter=_books_filter(),
        ),
    )


def test_search_matrix_pairs_selects_the_feature_slot():
    client, _folder = _ready()
    _books_selector(
        client,
        "search_matrix_pairs",
        lambda: client.search_matrix_pairs("items", query_filter=_books_filter()),
    )


def test_search_matrix_offsets_selects_the_feature_slot():
    client, _folder = _ready()
    _books_selector(
        client,
        "search_matrix_offsets",
        lambda: client.search_matrix_offsets("items", query_filter=_books_filter()),
    )


def test_upload_collection_upserts_on_the_feature_slot():
    client, _folder = _ready()
    calls = []
    _stub(client, calls, [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.upload_collection(
            "items",
            vectors=[[0.1, 0.2, 0.3, 0.4]],
            payload=[{"feature": "books"}],
            ids=[1],
            batch_size=1,
            parallel=2,
            wait=True,
        )
    upserts = [flat for name, flat in calls if name == "upsert"]
    assert len(upserts) == 1
    assert upserts[0]["shard_key_selector"] == "s0"
    assert any(item.category is ShardingWarning for item in caught)


def test_async_facet_selects_the_feature_slot():
    asyncio.run(_async_facet())


async def _async_facet():
    client, _folder = _ready(async_client=True)
    calls = []
    _stub_async(client, calls, [])
    await client.facet("items", key="feature", facet_filter=_books_filter())
    matched = [flat for name, flat in calls if name == "facet"]
    assert len(matched) == 1
    assert matched[0]["shard_key_selector"] == ["s0"]
