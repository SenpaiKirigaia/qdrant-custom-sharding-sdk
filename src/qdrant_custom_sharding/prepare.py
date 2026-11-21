import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from qdrant_client.models import VectorParams

from qdrant_custom_sharding.errors import ShardingError

_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".npz", ".parquet"}


def count_features(values: Iterable) -> dict[str, int]:
    """Count feature values. Map keys are strings, so int 7 becomes \"7\"."""
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            raise ValueError("feature value is missing")
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def vector_params(vectors, distance) -> VectorParams:
    """Read the vector width. distance is the one used when the vectors were built."""
    return VectorParams(size=_vector_size(vectors), distance=distance)


def collection_setup(path, feature_field: str, vector_field, distance) -> dict:
    """Scan a table once and return feature_counts and vectors_config.

    path is a .csv, .json, .jsonl, .ndjson, .npz, or .parquet file.
    vector_field is a column name, or a list of column names for a wide CSV.
    A single vector column holds a JSON list in CSV and a list in JSON.
    .parquet needs pyarrow.
    """
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(
            "path must end with csv, json, jsonl, ndjson, npz, or parquet"
        )
    counts: dict[str, int] = {}
    size: int | None = None
    for feature, vector in _rows(file, suffix, feature_field, vector_field):
        if feature is None:
            raise ValueError("feature value is missing")
        key = str(feature)
        counts[key] = counts.get(key, 0) + 1
        width = len(vector)
        if width == 0:
            raise ValueError("vector is empty")
        if size is None:
            size = width
        elif width != size:
            raise ValueError("vector width changed")
    if size is None:
        raise ValueError("file has no rows")
    return {
        "feature_counts": counts,
        "vectors_config": VectorParams(size=size, distance=distance),
    }


def _vector_size(vectors) -> int:
    if isinstance(vectors, np.ndarray):
        if vectors.size == 0 or vectors.ndim not in (1, 2):
            raise ValueError("vector input is empty or not a matrix")
        return int(vectors.shape[-1])
    rows = list(vectors)
    if not rows:
        raise ValueError("vector input is empty")
    first = rows[0]
    if isinstance(first, (int, float)):
        return len(rows)
    width = len(first)
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("vector rows have different widths")
    return width


def _rows(file: Path, suffix: str, feature_field: str, vector_field) -> Iterator:
    if suffix == ".csv":
        yield from _csv_rows(file, feature_field, vector_field)
    elif suffix == ".json":
        yield from _json_rows(file, feature_field, vector_field)
    elif suffix in {".jsonl", ".ndjson"}:
        yield from _jsonl_rows(file, feature_field, vector_field)
    elif suffix == ".npz":
        yield from _npz_rows(file, feature_field, vector_field)
    else:
        yield from _parquet_rows(file, feature_field, vector_field)


def _csv_rows(file: Path, feature_field: str, vector_field) -> Iterator:
    with file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row[feature_field], _csv_vector(row, vector_field)


def _csv_vector(row: dict, vector_field):
    if isinstance(vector_field, (list, tuple)):
        return [float(row[name]) for name in vector_field]
    return _as_vector(row[vector_field])


def _json_rows(file: Path, feature_field: str, vector_field) -> Iterator:
    rows = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("json file must be a list of objects")
    yield from _object_rows(rows, feature_field, vector_field)


def _jsonl_rows(file: Path, feature_field: str, vector_field) -> Iterator:
    if isinstance(vector_field, (list, tuple)):
        raise ValueError("a list of vector columns is only for csv")
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield row[feature_field], _as_vector(row[vector_field])


def _object_rows(rows, feature_field: str, vector_field) -> Iterator:
    if isinstance(vector_field, (list, tuple)):
        raise ValueError("a list of vector columns is only for csv")
    for row in rows:
        yield row[feature_field], _as_vector(row[vector_field])


def _npz_rows(file: Path, feature_field: str, vector_field) -> Iterator:
    if isinstance(vector_field, (list, tuple)):
        raise ValueError("a list of vector columns is only for csv")
    with np.load(file) as data:
        features = data[feature_field]
        vectors = data[vector_field]
        for feature, vector in zip(features, vectors, strict=True):
            yield feature.item() if hasattr(feature, "item") else feature, list(vector)


def _parquet_rows(file: Path, feature_field: str, vector_field) -> Iterator:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ShardingError("parquet needs pyarrow. Install pyarrow and run again.") from exc
    if isinstance(vector_field, (list, tuple)):
        columns = [feature_field, *vector_field]
    else:
        columns = [feature_field, vector_field]
    table = parquet.read_table(file, columns=columns)
    columns_dict = table.to_pydict()
    features = columns_dict[feature_field]
    if isinstance(vector_field, (list, tuple)):
        vectors = zip(*(columns_dict[name] for name in vector_field), strict=True)
    else:
        vectors = columns_dict[vector_field]
    for feature, vector in zip(features, vectors, strict=True):
        yield feature, _as_vector(vector)


def _as_vector(value):
    if isinstance(value, str):
        value = json.loads(value)
    return list(value)
