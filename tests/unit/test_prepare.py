import json

import numpy as np
import pytest
from qdrant_client.models import Distance, VectorParams

from qdrant_custom_sharding.errors import ShardingError
from qdrant_custom_sharding.prepare import (
    collection_setup,
    count_features,
    vector_params,
)


def test_count_features_uses_string_keys():
    assert count_features([7, 7, "8", np.int64(9)]) == {"7": 2, "8": 1, "9": 1}
    assert count_features([]) == {}


def test_count_features_rejects_a_missing_value():
    with pytest.raises(ValueError, match="feature"):
        count_features([7, None])


def test_vector_params_reads_the_width():
    matrix = [[0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.0, 0.2]]
    params = vector_params(matrix, Distance.COSINE)
    assert params == VectorParams(size=4, distance=Distance.COSINE)
    assert vector_params(np.zeros((3, 4)), Distance.DOT).size == 4
    assert vector_params([0.1, 0.2, 0.3, 0.4], Distance.COSINE).size == 4


def test_vector_params_rejects_empty_or_ragged_input():
    with pytest.raises(ValueError, match="vector"):
        vector_params([], Distance.COSINE)
    with pytest.raises(ValueError, match="vector"):
        vector_params([[0.1, 0.2], [0.1]], Distance.COSINE)


def test_collection_setup_reads_csv_json_and_jsonl(tmp_path):
    rows = [
        {"subject_id": 7, "vector": [0.1, 0.2, 0.3, 0.4]},
        {"subject_id": 7, "vector": [0.2, 0.2, 0.2, 0.2]},
        {"subject_id": 8, "vector": [0.0, 0.1, 0.0, 0.1]},
    ]
    csv_path = tmp_path / "people.csv"
    csv_path.write_text(
        "subject_id,vector\n"
        '7,"[0.1, 0.2, 0.3, 0.4]"\n'
        '7,"[0.2, 0.2, 0.2, 0.2]"\n'
        '8,"[0.0, 0.1, 0.0, 0.1]"\n',
        encoding="utf-8",
    )
    json_path = tmp_path / "people.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path = tmp_path / "people.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    expected = {
        "feature_counts": {"7": 2, "8": 1},
        "vectors_config": VectorParams(size=4, distance=Distance.COSINE),
    }
    assert collection_setup(csv_path, "subject_id", "vector", Distance.COSINE) == expected
    assert collection_setup(json_path, "subject_id", "vector", Distance.COSINE) == expected
    assert collection_setup(jsonl_path, "subject_id", "vector", Distance.COSINE) == expected


def test_collection_setup_reads_csv_vector_columns(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(
        "subject_id,v0,v1,v2,v3\n7,0.1,0.2,0.3,0.4\n8,0,0.1,0,0.1\n",
        encoding="utf-8",
    )
    setup = collection_setup(
        path, "subject_id", ["v0", "v1", "v2", "v3"], Distance.EUCLID
    )
    assert setup["feature_counts"] == {"7": 1, "8": 1}
    assert setup["vectors_config"].size == 4
    assert setup["vectors_config"].distance == Distance.EUCLID


def test_collection_setup_reads_npz(tmp_path):
    path = tmp_path / "people.npz"
    np.savez(
        path,
        subject_id=np.array([7, 7, 9]),
        vector=np.zeros((3, 4)),
    )
    setup = collection_setup(path, "subject_id", "vector", Distance.COSINE)
    assert setup["feature_counts"] == {"7": 2, "9": 1}
    assert setup["vectors_config"].size == 4


def test_collection_setup_rejects_an_unknown_suffix(tmp_path):
    path = tmp_path / "people.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="csv"):
        collection_setup(path, "subject_id", "vector", Distance.COSINE)


def test_collection_setup_parquet_needs_pyarrow(tmp_path):
    path = tmp_path / "people.parquet"
    path.write_bytes(b"")
    with pytest.raises(ShardingError, match="pyarrow"):
        collection_setup(path, "subject_id", "vector", Distance.COSINE)
