from datetime import datetime, timedelta, timezone

from qdrant_custom_sharding.map_cache import MapCache, Reason
from qdrant_custom_sharding.models import FeatureInfo, MapDoc
from tests.fakes import FakeMapStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _doc(version: int = 1) -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
        version=version,
    )


def _cache(store: FakeMapStore, ttl: int = 30):
    now = [NOW]

    def clock() -> datetime:
        return now[0]

    return MapCache(store, ttl, clock), now


def test_get_returns_a_fresh_cache_without_loading_again():
    store = FakeMapStore()
    store.save(_doc())
    store.loads = 0
    original = store.load

    def counting_load() -> MapDoc:
        store.loads += 1
        return original()

    store.load = counting_load
    cache, _now = _cache(store)
    first, first_reason = cache.get()
    second, second_reason = cache.get()
    assert first_reason is None
    assert second_reason is None
    assert second == first
    assert store.loads == 1


def test_get_reloads_after_the_ttl():
    store = FakeMapStore()
    store.save(_doc())
    store.loads = 0
    original = store.load

    def counting_load() -> MapDoc:
        store.loads += 1
        return original()

    store.load = counting_load
    cache, now = _cache(store, ttl=30)
    cache.get()
    now[0] = NOW + timedelta(seconds=29)
    cache.get()
    assert store.loads == 1
    now[0] = NOW + timedelta(seconds=30)
    loaded, reason = cache.get()
    assert reason is None
    assert loaded == _doc()
    assert store.loads == 2


def test_get_returns_stale_map_when_reload_fails():
    store = FakeMapStore()
    store.save(_doc())
    cache, now = _cache(store, ttl=30)
    cached, reason = cache.get()
    assert reason is None
    now[0] = NOW + timedelta(seconds=30)
    store.fail_with_unavailable = True
    loaded, reason = cache.get()
    assert reason == Reason.stale_map
    assert loaded == cached


def test_get_returns_no_map_when_unavailable_and_empty():
    store = FakeMapStore()
    store.fail_with_unavailable = True
    cache, _now = _cache(store)
    assert cache.get() == (None, Reason.no_map)


def test_reason_values():
    assert Reason.stale_map == "stale_map"
    assert Reason.no_map == "no_map"
    assert [item.value for item in Reason] == ["stale_map", "no_map"]


def test_get_returns_no_map_when_the_map_is_missing():
    store = FakeMapStore()
    store.loads = 0
    original = store.load

    def counting_load() -> MapDoc:
        store.loads += 1
        return original()

    store.load = counting_load
    cache, now = _cache(store, ttl=30)
    assert cache.get() == (None, None)
    store.save(_doc())
    assert cache.get() == (None, None)
    assert store.loads == 1
    now[0] = NOW + timedelta(seconds=30)
    loaded, reason = cache.get()
    assert reason is None
    assert loaded == _doc()


def test_unavailable_with_nothing_cached_is_not_stored():
    store = FakeMapStore()
    store.fail_with_unavailable = True
    cache, _now = _cache(store)
    assert cache.get() == (None, Reason.no_map)
    store.fail_with_unavailable = False
    store.save(_doc())
    loaded, reason = cache.get()
    assert reason is None
    assert loaded == _doc()


def test_forget_drops_a_cached_map_and_a_missing_map():
    store = FakeMapStore()
    store.save(_doc())
    cache, _now = _cache(store)
    cache.get()
    cache.forget()
    store.fail_with_unavailable = True
    assert cache.get() == (None, Reason.no_map)

    empty = FakeMapStore()
    missing, _clock = _cache(empty)
    assert missing.get() == (None, None)
    missing.forget()
    empty.save(_doc())
    loaded, reason = missing.get()
    assert reason is None
    assert loaded == _doc()


def test_expire_reloads_and_keeps_the_old_map_when_the_load_fails():
    store = FakeMapStore()
    store.save(_doc())
    cache, _now = _cache(store, ttl=30)
    cached, reason = cache.get()
    assert reason is None
    cache.expire()
    store.fail_with_unavailable = True
    loaded, reason = cache.get()
    assert reason == Reason.stale_map
    assert loaded == cached


def test_put_serves_the_written_map_until_the_ttl():
    store = FakeMapStore()
    store.save(_doc())
    cache, now = _cache(store, ttl=30)
    cache.get()
    replacement = _doc(version=2)
    cache.put(replacement)
    store.fail_with_unavailable = True
    loaded, reason = cache.get()
    assert reason is None
    assert loaded == replacement
    now[0] = NOW + timedelta(seconds=30)
    stale, reason = cache.get()
    assert reason == Reason.stale_map
    assert stale == replacement
