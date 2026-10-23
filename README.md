# qdrant-custom-sharding-sdk

Russian: [README.ru.md](README.ru.md).

Install the package `qdrant-custom-sharding-sdk`. Import `qdrant_custom_sharding`.

## What it is

`ShardedQdrantClient` is a subclass of `QdrantClient`. You change the class name, pass a map folder, and keep the rest of the calls. For a collection that has a map, the client fills in `shard_key_selector` from the feature in your filter or payload. A collection with no map is left alone and behaves like a normal `QdrantClient`.

The feature is any payload field you choose per collection, for example `subject_id`. Values are `int` or `str`. The client packs thousands of those values into at most 48 slots. A slot is one Qdrant shard key. The map that says which value lives on which slot is one JSON file per collection, in a local folder or under an S3 or MinIO prefix.

## Why

A normal Qdrant collection keeps every point on every shard. A search, a count, or an update then waits for all of the shards, even when you only wanted one `subject_id`. If almost every call stays inside one feature, that fan-out is the slow part.

Custom sharding can send the call to one shard key, but Qdrant only wants a small number of keys. One key per `subject_id` does not fit thousands of ids, and a few large subjects would still land on one shard and overload it. This SDK keeps a map from each feature value to one or more of at most 48 slots. Many small values share a slot. A large value can take several slots, and the client picks among them with `crc32` of the point id.

You do not pass `shard_key_selector` yourself. A `must` filter on the feature field is enough, and that filter is the same one you already write to search inside one subject. The client reads the value, looks it up, and adds the slot. A filter that does not name the feature still reads every shard, and the client emits `FanOutWarning`, because it cannot tell which slot is enough.

## Install

You need Python 3.11 or newer, `qdrant-client` 1.12 or newer, and Qdrant server 1.12 or newer. The package also installs `boto3` 1.36 or newer. There is no upper pin on `qdrant-client`, because the client overrides public methods of `QdrantClient`: a newer release can remove or change a method, and then this client lacks it too. Pin a tested version in production. The tested clients are 1.19.1 and 1.12.2. The tested server is 1.19.2.

The package is not on PyPI. Install it from a local copy, or from git if this repository has a remote.

```text
uv add path/to/qdrant-custom-sharding-sdk
pip install path/to/qdrant-custom-sharding-sdk
uv add git+<url>
pip install git+<url>
```

In this repository, `uv sync` and `make install` do the same thing.

## Quick start

Replace `QdrantClient` with `ShardedQdrantClient`, pass `url` and `map_uri`, and create the collection with `client.sharding.create_collection`. After that, `upsert`, `query_points`, `count`, and `scroll` stay ordinary Qdrant calls. Put the feature field in the filter so the client can choose a slot.

A plain `create_collection` does not write a map, so the client will not route that collection. `client.sharding.create_collection` creates the collection, the slots, the payload index, and the map. Slot names run from `s0` upward, one name per slot, up to `n_slots`. Keys in `feature_counts` are strings, so the integer 7 is the key `"7"`. For `feature_type="keyword"` the index is a tenant index. For `feature_type="integer"` the index is a lookup index, and range is off.

A point with no feature field raises `ValueError`. An upsert of a value the map has never seen adds that value, then writes the points.

You do not have to count features by hand. `collection_setup` reads a CSV, JSON, JSONL, NumPy `.npz`, or Parquet file and returns `feature_counts` and `vectors_config`. The distance is still yours, because the file does not know if the vectors were built for cosine or dot. Parquet needs `pyarrow`. A NumPy array or a memmap that you already have in memory uses `count_features` and `vector_params` instead. The notebook [examples/new_collection.ipynb](examples/new_collection.ipynb) shows each format.

```python
from qdrant_client.models import Distance

from qdrant_custom_sharding import collection_setup

setup = collection_setup("people.csv", "subject_id", "vector", Distance.COSINE)
client.sharding.create_collection(
    "people",
    "subject_id",
    feature_type="integer",
    n_slots=48,
    **setup,
)
```

In a CSV, the `vector` column is a JSON list. A wide CSV can pass the column names instead: `["v0", "v1", "v2", "v3"]`.

### Integer feature

```python
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from qdrant_custom_sharding import ShardedQdrantClient

client = ShardedQdrantClient(
    url="http://localhost:6333",
    map_uri="C:/qdrant-maps",
)

client.sharding.create_collection(
    "people",
    "subject_id",
    feature_type="integer",
    n_slots=48,
    feature_counts={"7": 10000, "8": 40, "9": 40},
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)

client.upsert(
    collection_name="people",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"subject_id": 7},
        )
    ],
)

query_filter = Filter(
    must=[
        FieldCondition(key="subject_id", match=MatchValue(value=7)),
    ]
)

hits = client.query_points(
    collection_name="people",
    query=[0.1, 0.2, 0.3, 0.4],
    query_filter=query_filter,
    limit=10,
)

total = client.count(
    collection_name="people",
    count_filter=query_filter,
    exact=True,
)

records, next_offset = client.scroll(
    collection_name="people",
    scroll_filter=query_filter,
    limit=64,
)
```

### String feature

```python
client.sharding.create_collection(
    "catalog",
    "category",
    feature_type="keyword",
    n_slots=12,
    feature_counts={"books": 2000, "maps": 30},
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)

client.upsert(
    collection_name="catalog",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"category": "books"},
        )
    ],
)

books = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="books")),
    ]
)

hits = client.query_points(
    collection_name="catalog",
    query=[0.1, 0.2, 0.3, 0.4],
    query_filter=books,
    limit=10,
)
```

## Async client

`AsyncShardedQdrantClient` is the same idea for `AsyncQdrantClient`. The constructor keywords match the parent. Admin calls are `await client.sharding.<method>(...)`. The map stores are synchronous, and boto3 or a local file would block the event loop, so the async client runs those reads on a worker thread with `asyncio.to_thread`.

```python
import asyncio

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from qdrant_custom_sharding import AsyncShardedQdrantClient

async def main():
    client = AsyncShardedQdrantClient(
        url="http://localhost:6333",
        map_uri="C:/qdrant-maps",
    )
    try:
        await client.sharding.create_collection(
            "catalog",
            "category",
            feature_type="keyword",
            n_slots=12,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        await client.upsert(
            collection_name="catalog",
            points=[
                PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"category": "books"},
                )
            ],
            wait=True,
        )
        books = Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value="books")),
            ]
        )
        hits = await client.query_points(
            collection_name="catalog",
            query=[0.1, 0.2, 0.3, 0.4],
            query_filter=books,
            limit=10,
        )
    finally:
        await client.close()

asyncio.run(main())
```

## Editor support

Completion, hover, and signatures follow `QdrantClient` or `AsyncQdrantClient` for the `qdrant-client` you have installed. The package ships `.pyi` stubs and `py.typed`. Routed methods are not repeated in the stub, so the type checker inherits them from the parent, and `help(client.upsert)` still shows the Qdrant signature because each override uses `functools.wraps`. After you install or upgrade, restart the language server.

The stub copies the constructor parameter list from `qdrant-client` 1.19.1. On another client version, constructor hover can differ from the constructor you actually call.

## Configuration

| Argument | Default | Effect |
|---|---|---|
| `map_uri` | env `MAP_URI`, or none | folder of map files |
| `map_ttl_seconds` | 30 | seconds before the next load |
| `sharding_warnings` | `True` | emit warnings |
| `detect_feature_changes` | `True` | read the old value before upsert |

`map_folder`, `now_fn`, and `sleep_fn` exist so tests can inject a folder and a clock. With no `map_uri` and no `MAP_URI`, nothing is sharded, and admin calls raise `ValueError` with the text `no map folder`.

### Map folder

The URI is a folder, not one file. Each collection is `<folder>/<collection>.json`. Use one folder for one Qdrant instance.

- A folder path: `C:/qdrant-maps` or `C:\qdrant-maps`
- A file URI: `file:///C:/qdrant-maps/`
- S3 or MinIO: `s3://bucket/prefix/`

### S3

boto3 reads credentials from the environment. Do not pass keys into the client.

| Variable | Role |
|---|---|
| `AWS_ENDPOINT_URL` | custom endpoint, for example MinIO |
| `AWS_ACCESS_KEY_ID` | access key |
| `AWS_SECRET_ACCESS_KEY` | secret key |
| `AWS_DEFAULT_REGION` | region |

The SDK reads the region from `AWS_DEFAULT_REGION` or from `AWS_REGION`. A custom endpoint with no region uses `us-east-1`. With no `AWS_ENDPOINT_URL`, boto3 uses its normal AWS settings.

`boto3` 1.36 computes checksums by default, which breaks some S3-compatible stores, so the SDK sets `request_checksum_calculation` and `response_checksum_validation` to `when_required`. Map saves are conditional: the first save sends `If-None-Match: *`, and a later save sends `If-Match` with the current ETag. AWS S3 and MinIO both accept those headers. A missing access key or secret key is a setup error. botocore raises it, and the SDK does not catch it.

PowerShell for the local MinIO stack:

```powershell
$env:MAP_URI = "s3://sharding-maps/app/"
$env:AWS_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin123"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

## How routing works

If you pass `shard_key_selector` yourself, the call goes to Qdrant unchanged. The same happens when the collection has no map. Otherwise the client picks slots from the feature.

Only a `must` condition selects the feature, including a `Filter` nested in `must`. The client reads `MatchValue` and `MatchAny`. A range match does not count, and neither do `should`, `must_not`, `min_should`, or a `NestedCondition`. Those calls fan out.

Reads:

| Method | Rule |
|---|---|
| `query_points` | by `query_filter` |
| `query_points_groups` | by `query_filter` |
| `count` | by `count_filter` |
| `scroll` | by `scroll_filter` |
| `facet` | by `facet_filter` |
| `search_matrix_pairs` | by `query_filter` |
| `search_matrix_offsets` | by `query_filter` |
| `query_batch_points` | by the filter on each request |

`query_batch_points` sets `shard_key` on a copy of each request. A request that already has `shard_key` stays unchanged.

Writes:

| Method | Rule |
|---|---|
| `upsert` | by the feature in the payload |
| `upload_points` | through `upsert` |
| `upload_collection` | through `upsert` |
| `delete` | feature filter: both copies. Otherwise every slot |
| `set_payload` | feature filter, or a safe move. No feature: every slot, or the point slot for ids |
| `overwrite_payload` | safe move when the payload has the feature |
| `delete_payload` | feature filter: both copies. No feature: every slot, or the point slot for ids |
| `delete_vectors` | feature filter: both copies. No feature: every slot, or the point slot for ids |
| `batch_update_points` | each operation, in order |

Other calls:

| Method | Rule |
|---|---|
| `retrieve` | no shard key, `FanOutWarning` |
| `update_vectors` | the slot that holds each id, `FanOutWarning` |
| `delete_collection` | then delete the map |

An upsert of an unknown feature adds it and writes its slot. An id update loads the map once. If the feature is still unknown, or the stored point has no feature field, the update uses the slot on the retrieved record, one shard key per call. A missing id goes to the first slot and returns 404. When the record has no shard key, the client tries each slot on its own, and a 404 from an empty slot does not cancel the other slots.

These calls emit `FanOutWarning` because they do not name one feature:

- `retrieve` has no shard key, so Qdrant reads all shards
- `update_vectors` uses the slot that holds each id
- `delete` by id, or by a filter with no feature, uses every slot
- `set_payload`, `delete_payload`, and `delete_vectors` by id use the point's slot
- those three by a filter with no feature use every slot
- `NestedCondition`, `should`, `must_not`, `min_should`, and an unknown value after one map reload

When a map exists and you do not pass `shard_key_selector`, `clear_payload` raises `ValueError`. So does `delete_payload` when `keys` include the feature field, and `overwrite_payload` when the payload omits the feature field. The point would have no feature left.

`delete`, `set_payload`, `delete_payload`, and `delete_vectors` write both `shard_keys` and `write_to` when the filter names the feature, so both copies change during a move, including `cutover`. Partial updates are `set_payload`, `overwrite_payload`, `delete_payload`, `update_vectors`, and `delete_vectors`. `delete` does not copy the point first.

| State | Id update |
|---|---|
| `stable` | the old slot |
| `copying` | copy the point onto the new slot, then update both slots |
| `dual` | both slots |
| `cutover` | the new slot |

During `copying`, a filter that names that feature copies the matching points first. A filter with no feature does the same when any feature is `copying`. The filter write then uses the same slots as before.

Two clients can write different values to one point at the same time, and the two copies can apply those writes in a different order. The SDK does not override `search`, `recommend`, or `discover`. Those methods exist only on old clients. Qdrant fans them out, and the results stay correct.

### Warnings

`ShardingWarning` subclasses `UserWarning`. `FanOutWarning` means the call did not name one feature, so the client does extra work. `FeatureMovedWarning` means points moved to a new feature value. `StaleMapWarning` means the client kept a cached map, or a custom collection had no map.

Pass `sharding_warnings=False` to silence them, or ignore a category with `warnings.filterwarnings`. Ignore `ShardingWarning` to silence every SDK warning, or ignore one subclass to silence only that warning.

```python
import warnings

from qdrant_custom_sharding import ShardingWarning

warnings.filterwarnings("ignore", category=ShardingWarning)
```

### Feature value changes

`upsert`, `set_payload`, and `overwrite_payload` can change the feature on a point. The SDK writes the point to the new slots with `wait=True`, then deletes it on the old slots with `wait=True`. A crash can leave a second copy, but the first copy remains, so the point is not lost. The client emits one `FeatureMovedWarning` with the moved-point count.

On upsert, the client reads the old payload when `detect_feature_changes` is true. Set it to `False` to skip that read. `set_payload` and `overwrite_payload` ignore the flag. A `set_payload` call with `key` set does not move the point.

## Admin

Map operations live on `client.sharding`. SDK errors subclass `ShardingError`. Two admin writes can conflict. The SDK retries the save five times, then raises `VersionConflict`. Upsert does not update the counts in the map. Call `refresh_stats` when you want counts from Qdrant.

Do not pass `sharding_method` or `shard_number`. The SDK sets custom sharding and `shard_number` 1. Other arguments go to Qdrant `create_collection`, for example `vectors_config`. `feature_type` is `keyword` or `integer`.

- `create_collection(collection_name, feature_field, feature_type="keyword", n_slots=48, feature_counts=None, **kwargs)` creates the collection, the slots, the index, and the map.
- `add_feature(collection, value)` places the value on the least loaded slot.
- `migrate(collection, value, to_slot, grace_seconds=60)` moves the feature onto one slot and waits `grace_seconds`.
- `split(collection, value, partitions, grace_seconds=60)` spreads the feature across `partitions` free slots.
- `start_move(collection, value, targets, grace_seconds=60)` changes `stable` to `copying`.
- `copy_feature(collection, value, batch=256)` copies points onto `write_to` and returns the point count.
- `open_dual(collection, value, grace_seconds=60)` changes `copying` to `dual`.
- `cutover(collection, value, grace_seconds=60)` changes `dual` to `cutover`. It returns a `CutoverResult` with `old_slots`, `copied`, and `removed`.
- `finish_move(collection, value)` deletes the feature on the old slots, then changes `cutover` to `stable`.
- `alias_feature(collection, old, new, grace_seconds=60)` renames the map key. See Aliases.
- `drop_alias(collection, old)` rewrites points that still use `old`, then removes that alias.
- `refresh_stats(collection)` stores `points_count` and `slot_load` from Qdrant.
- `get_map(collection)` returns the map, or `None` when no map file exists.

### Move states

The client picks a slot with `crc32` of `str(point_id)`. `shard_keys` is where readers look while the state is `stable` or `copying`. `write_to` is the target list during a move.

| State | Write slots | Read slots |
|---|---|---|
| `stable` | one slot from `shard_keys` | `shard_keys` |
| `copying` | the old slot and the new slot | `shard_keys` |
| `dual` | the old slot and the new slot | `write_to` |
| `cutover` | one slot from `write_to` | `write_to` |

In `copying` and `dual` the two picks can name the same slot. A read uses one placement, so `count`, `limit`, and `offset` stay exact. During `copying` and `dual`, each upsert writes the old slot and the new slot, so a point that arrives in the middle of the move is not left only on the old slot.

`grace_seconds` must be longer than `map_ttl_seconds` on every reader. The defaults are 60 and 30. A reader that still has the previous map then hits a placement that is still complete. Wait for `grace_until` before `open_dual`, `cutover`, and `finish_move`. An early call raises `MoveNotReady`. A wrong state raises `InvalidMove`.

Every target must already be a slot on the map. Targets must not overlap `shard_keys`, must not be repeated, and must not be busy. A busy slot is already in a move.

### Migrate and split

`migrate` sleeps for `grace_seconds` three times: before the copy, before the read switch, and before the delete. A call with `grace_seconds` 60 therefore waits about three minutes, and other clients can upsert during the waits. Pick a target slot that the feature does not already use.

```python
doc = client.sharding.get_map("people")
current = doc.features["7"].shard_keys
target = next(slot for slot in doc.slots if slot not in current)
client.sharding.migrate("people", 7, target, grace_seconds=60)
```

`split` chooses the slots for you. A free slot is not busy and is not in the current `shard_keys`. If free slots are fewer than `partitions`, `split` raises `InvalidMove`. The waits are the same as `migrate`. For a string feature, pass a str value, for example `"books"`.

```python
client.sharding.split("people", 8, 3, grace_seconds=60)
doc = client.sharding.refresh_stats("people")
print(doc.slot_load)
print(doc.features["8"].points_count)
print(doc.features["8"].shard_keys)
```

### Aliases

`alias_feature(collection, old, new, grace_seconds=60)` renames the map key from `old` to `new` and keeps `old` in `aliases`. It waits `grace_seconds`, which must be longer than `map_ttl_seconds` on every reader, then sets the feature field on stored points to `new`. The return value is how many points changed.

Until that rewrite finishes, a filter on `new` finds only points written after the rename. After the rewrite, a filter on `old` finds only points written later with `old`. Those points stay on the same slots.

`drop_alias(collection, old)` runs the same rewrite for points that still use `old`, then removes `old` from `aliases`, and returns the count. After that, a write that uses `old` is a new feature. Both calls require state `stable`. A running move raises `InvalidMove`.

## Map outage

If a reload fails and a cached map exists, the client uses that map and emits `StaleMapWarning`. If nothing is cached, it calls `get_collection` once and stores the sharding method. A custom collection then fans reads out, with `StaleMapWarning`, and a write raises `WriteDenied`. Writes here include `upsert`, `delete`, `set_payload`, and `update_vectors`. `retrieve` is a read, so it fans out during the outage. An auto collection is sent through unchanged, because it was never sharded by this map.

The S3 connect timeout is 2 seconds and the read timeout is 5 seconds, with at most two tries, so a cold load with S3 down fails in about 5 seconds. `delete_collection` deletes the Qdrant collection first. On S3, a failed map delete raises `MapUnavailable`.

## Limits

`upload_points` and `upload_collection` with `parallel` greater than 1 run one batch after another and emit `ShardingWarning`. That path ignores `max_retries` and `method`. A collection with no map still uses the native Qdrant upload.

`search` and `recommend` are not routed. They exist only in old `qdrant-client` versions. Qdrant sends them to all shards, and the results stay correct.

A `NestedCondition` filter fans out. Only `must` conditions select a feature, and only `MatchValue` and `MatchAny`. A range match does not. A collection has one feature field.

The recommended maximum is 48 slots, which is also the default `n_slots`. The SDK does not reject a larger number. A move needs `qdrant-client` and Qdrant server 1.17 or newer, because insert-only upserts start there. An older client raises `ShardingError` on copy.

A delete and a write of the same point by different clients, within milliseconds during a move, can leave the copies different until the reconcile before cutover. Two clients can also write different values to the same point at the same time, and the two copies can apply those writes in a different order.

## Local development

Install Docker Desktop, uv, and GNU make.

From the repository root, `make up` starts three Qdrant peers and MinIO. The peers are one cluster. Call port 6333. REST ports are 6333, 6433, and 6533. gRPC ports are 6334, 6434, and 6534. The Qdrant image is `qdrant/qdrant:v1.19.2`.

MinIO listens on port 9000, and the console on port 9001. The bucket is `sharding-maps`. The user is `minioadmin` and the password is `minioadmin123`. The default image comes from quay.io. To use this MinIO from your code, set the variables in Configuration. Integration tests set `AWS_ENDPOINT_URL` and the MinIO keys themselves.

| Target | What it does |
|---|---|
| `install` | sync dev dependencies with uv |
| `test` | unit tests only |
| `test-integration` | pytest integration marker |
| `test-e2e` | pytest e2e marker |
| `test-all` | unit, integration, and e2e in one run |
| `test-min-client` | unit and integration with qdrant-client 1.12 |
| `typecheck` | mypy and pyright reveal of client method signatures |
| `up` | start docker compose stack |
| `down` | stop stack, keep volumes |
| `reset` | stop stack and remove volumes |
| `status` | docker compose ps |
| `logs` | last 100 log lines |

`make test` runs unit tests with fakes and does not need Docker. `make test-integration` calls real Qdrant and MinIO, for both the sync and the async client. `make test-e2e` loads a skewed int collection and a skewed str collection. `make test-min-client` runs unit and integration tests with `qdrant-client` 1.12 and skips e2e. `make typecheck` runs mypy and pyright on the client method signatures.

If quay.io denies the pull, set `MINIO_IMAGE` to another MinIO image. A locally cached `minio/minio` tag works. Then run `make up` again.

A local HTTP proxy on Windows resets connections to localhost. Set `NO_PROXY` to `localhost,127.0.0.1`, or pass `trust_env=False` to the constructor.

If Docker reports `Internal Server Error for API route`, stop Docker Desktop, run `wsl --shutdown` from PowerShell, and start Docker Desktop again.

## Design notes

A map save compares the stored version and writes only on a match. A mismatch raises `VersionConflict`. S3 sends `If-Match`, or `If-None-Match` on the first save. A MongoDB update can do the same with a version field. See [write atomicity](https://www.mongodb.com/docs/manual/core/write-operations-atomicity/).

Placement takes the largest features first. This is longest-processing-time-first scheduling. See [the description](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling). The target is the total point count divided by the slot count, rounded up. A feature larger than the target takes more than one slot, and its points go to a slot by `crc32` of the point id.
