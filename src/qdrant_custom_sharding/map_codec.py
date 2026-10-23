import json
from datetime import datetime, timezone

from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc


def map_to_json(doc: MapDoc) -> str:
    payload = {
        "collection": doc.collection,
        "feature_field": doc.feature_field,
        "feature_type": doc.feature_type,
        "slots": list(doc.slots),
        "features": {
            name: _feature_to_json(info) for name, info in doc.features.items()
        },
        "slot_load": doc.slot_load,
        "version": doc.version,
    }
    return json.dumps(payload, indent=2)


def map_from_json(text: str) -> MapDoc:
    payload = json.loads(text)
    features = {
        name: _feature_from_json(row) for name, row in payload["features"].items()
    }
    return MapDoc(
        collection=payload["collection"],
        feature_field=payload["feature_field"],
        slots=list(payload["slots"]),
        feature_type=str(payload.get("feature_type", "keyword")),
        features=features,
        slot_load={
            str(slot): int(count) for slot, count in payload["slot_load"].items()
        },
        version=int(payload["version"]),
    )


def _feature_to_json(info: FeatureInfo) -> dict[str, object]:
    write_to = None if info.write_to is None else list(info.write_to)
    return {
        "shard_keys": list(info.shard_keys),
        "state": info.state.value,
        "write_to": write_to,
        "points_count": info.points_count,
        "aliases": list(info.aliases),
        "grace_until": _time_to_json(info.grace_until),
        "copied": info.copied,
    }


def _feature_from_json(row: dict[str, object]) -> FeatureInfo:
    write_to = row["write_to"]
    aliases = row["aliases"]
    shard_keys = row["shard_keys"]
    return FeatureInfo(
        shard_keys=list(shard_keys),
        state=FeatureState(str(row["state"])),
        write_to=None if write_to is None else list(write_to),
        points_count=int(row["points_count"]),
        aliases=list(aliases),
        grace_until=_time_from_json(row["grace_until"]),
        copied=bool(row.get("copied", False)),
    )


def _time_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("grace_until needs a timezone")
    text = value.astimezone(timezone.utc).isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _time_from_json(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
