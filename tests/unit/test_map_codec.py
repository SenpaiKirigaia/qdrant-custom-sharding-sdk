from datetime import UTC, datetime

import pytest

from qdrant_custom_sharding.map_codec import map_from_json, map_to_json
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc


def test_map_json_round_trip_keeps_utc_datetime_and_enums():
    grace = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)
    doc = MapDoc(
        collection="items",
        feature_field="feature",
        feature_type="integer",
        slots=["s0", "s1"],
        features={
            "books": FeatureInfo(
                shard_keys=["s0"],
                state=FeatureState.dual,
                write_to=["s1"],
                points_count=12,
                aliases=["1"],
                grace_until=grace,
                copied=True,
            ),
            "toys": FeatureInfo(shard_keys=["s0", "s1"]),
        },
        slot_load={"s0": 10, "s1": 2},
        version=4,
    )
    text = map_to_json(doc)
    assert "dual" in text
    assert "2026-08-22T15:00:00Z" in text
    loaded = map_from_json(text)
    assert loaded == doc
    assert loaded.features["books"].state is FeatureState.dual
    assert loaded.features["books"].grace_until == grace
    assert loaded.features["books"].copied is True
    assert loaded.features["toys"].copied is False
    assert loaded.features["toys"].write_to is None
    assert loaded.features["toys"].state is FeatureState.stable
    assert loaded.feature_type == "integer"


def test_map_to_json_rejects_a_naive_grace_time():
    doc = MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={
            "books": FeatureInfo(
                shard_keys=["s0"],
                grace_until=datetime(2026, 8, 22, 15, 0),
            ),
        },
        slot_load={"s0": 0},
        version=1,
    )
    with pytest.raises(ValueError, match="timezone"):
        map_to_json(doc)


def test_map_from_json_without_feature_type_is_keyword():
    text = """
    {
      "collection": "items",
      "feature_field": "feature",
      "slots": ["s0"],
      "features": {},
      "slot_load": {"s0": 0},
      "version": 1
    }
    """
    assert map_from_json(text).feature_type == "keyword"


def test_map_from_json_accepts_numeric_offset():
    text = """
    {
      "collection": "items",
      "feature_field": "feature",
      "slots": ["s0"],
      "features": {
        "books": {
          "shard_keys": ["s0"],
          "state": "cutover",
          "write_to": ["s0"],
          "points_count": 1,
          "aliases": [],
          "grace_until": "2026-08-22T15:00:00+00:00"
        }
      },
      "slot_load": {"s0": 1},
      "version": 2
    }
    """
    doc = map_from_json(text)
    assert doc.version == 2
    assert doc.features["books"].state is FeatureState.cutover
    assert doc.features["books"].grace_until == datetime(
        2026, 8, 22, 15, 0, tzinfo=UTC
    )
    assert doc.features["books"].copied is False
