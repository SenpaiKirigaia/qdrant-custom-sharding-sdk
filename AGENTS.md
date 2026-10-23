# Agent map

This repository is a drop-in client for a Qdrant collection that uses custom sharding. `ShardedQdrantClient` is a subclass of `QdrantClient`. The client routes a collection when its map file exists. The map path is `<map folder>/<collection>.json`. A collection with no map uses the parent client as it is.

`docs/architecture.md` is the build contract. `README.md` is the caller guide for install, routing, admin, and limits.

A feature is the payload value that selects a bucket. A slot is a shard key. Use those two words.

## First files

| Path | Role |
|---|---|
| `src/qdrant_custom_sharding/sharded_client.py` | Sync client, `_CallRouter`, `ShardingAdmin` |
| `src/qdrant_custom_sharding/async_sharded_client.py` | Async client, same pure router |
| `src/qdrant_custom_sharding/models.py` | `FeatureState`, `FeatureInfo`, `MapDoc`, no I/O |
| `src/qdrant_custom_sharding/routing.py` | `pick_slot`, `write_slots`, `read_slots` |
| `src/qdrant_custom_sharding/router.py` | `plan_writes`, `read_selector` |
| `src/qdrant_custom_sharding/planning.py` | `plan_placement` |
| `src/qdrant_custom_sharding/services.py` | Sync admin steps. Each function takes ports. |
| `src/qdrant_custom_sharding/async_services.py` | Async admin steps that call Qdrant. |
| `src/qdrant_custom_sharding/qdrant_store.py` | Qdrant adapter. Each call gets a shard key. |
| `src/qdrant_custom_sharding/map_stores.py` | File map store and S3 map store. |
| `src/qdrant_custom_sharding/map_cache.py` | Map cache, TTL, stale fallback |
| `src/qdrant_custom_sharding/errors.py` | `WriteDenied`, `VersionConflict`, `MoveNotReady`, `InvalidMove` |
| `src/qdrant_custom_sharding/sharding_warnings.py` | `FanOutWarning`, `FeatureMovedWarning`, `StaleMapWarning` |
| `src/qdrant_custom_sharding/prepare.py` | `collection_setup`, `count_features`, `vector_params` |
| `src/qdrant_custom_sharding/sharded_client.pyi` | Public stub, Pyright inherits Qdrant methods from it |
| `tests/fakes.py` | Fakes for unit tests |

The async stub is `src/qdrant_custom_sharding/async_sharded_client.pyi`.

## Caller surface

`__all__` in `src/qdrant_custom_sharding/__init__.py` is the import list. The client classes are `ShardedQdrantClient` and `AsyncShardedQdrantClient`. Admin calls use `client.sharding`.

`ShardingAdmin` and `AsyncShardingAdmin` expose these methods:

- `create_collection`
- `add_feature`
- `get_map`
- `read_load`
- `refresh_stats`
- `add_slot`
- `plan_rebalance`
- `apply_rebalance`
- `resume_move`
- `migrate`
- `split`
- `start_move`
- `copy_feature`
- `open_dual`
- `cutover`
- `finish_move`
- `alias_feature`
- `drop_alias`

## Invariants

Domain modules do no I/O. `services.py` and `async_services.py` take ports. Unit tests use `tests/fakes.py`.

`MapDoc` methods do not change `version`. `update_map` in `services.py` adds 1 and saves with a version check.

`read_load` returns the per-slot feature counts from Qdrant. It does not save the map. `refresh_stats` is the only writer of point counts in the map. An upsert does not update `slot_load`.

`plan_rebalance` uses that same read. The planner keeps a same-width move only when the fullest slot gets smaller. `apply_rebalance` runs that plan. `resume_move` continues an open move. `migrate` calls it.

A move uses this order. `stable`, then `copying`, then `dual`, then `cutover`, then `stable`. `migrate` runs that sequence. Do not skip a state.

`grace_seconds` must be greater than the `map_ttl_seconds` of every reader. The default grace is 60 seconds. The default TTL is 30 seconds.

A call that already has `shard_key_selector` goes to Qdrant unchanged. Admin code uses that path.

`search` and `recommend` are not routed. `query_points` is the routed query.

The install floor is `qdrant-client` 1.12. A move needs `qdrant-client` 1.17 and Qdrant server 1.17. Insert-only upsert starts in that release. An older client raises `ShardingError` on copy.

After a change to the `QdrantClient` constructor, update `__init__` in both stub files.

## Commands

Run commands from the repository root. `make test` does not need Docker. `make test-integration` and `make test-e2e` need `make up` first.

| Command | Effect |
|---|---|
| `make install` | Sync dev dependencies with uv |
| `make test` | Unit tests |
| `make test-integration` | Pytest integration marker |
| `make test-e2e` | Pytest e2e marker |
| `make test-all` | Unit, integration, and e2e |
| `make test-min-client` | Unit and integration on qdrant-client 1.12 |
| `make typecheck` | Mypy and pyright on client signatures |
| `make up` | Start three Qdrant peers and MinIO |
| `make down` | Stop the stack, volumes stay |
| `make reset` | Stop the stack and delete volumes |
| `make load` | Upsert points, then migrate while readers run |

`pyproject.toml` skips markers `integration`, `e2e`, and `typing` on a plain pytest run.

On Windows, a local HTTP proxy can reset a connection to localhost. Set `NO_PROXY` to `localhost,127.0.0.1` before an integration test.

## Tests

| Path | What it checks |
|---|---|
| `tests/unit/` | Pure modules, the cache, and services on fakes |
| `tests/integration/` | Real Qdrant and MinIO, sync and async |
| `tests/e2e/test_skewed_collection.py` | Skewed int collection and skewed str collection |
| `tests/typing/` | Pyright visibility of public methods |

## Edits

Put a new routing rule in `routing.py` or `router.py`. The client applies the plan. It does not own the rule.

Put a new admin step in `services.py` as a function that takes ports. Call it from `ShardingAdmin`. Add the async twin in `async_services.py` and `AsyncShardingAdmin`.

Add a unit test on fakes before an integration test. Use an integration test only for Qdrant behavior that a fake cannot show.

Do not put secrets in docs or in tests. The local MinIO user and password are already in `README.md`.
