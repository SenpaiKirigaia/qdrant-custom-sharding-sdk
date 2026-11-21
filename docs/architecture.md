# Architecture

Build contract for the code. Pattern: ports and adapters plus a service layer, as in Percival and Gregory, *Architecture Patterns with Python*. The domain has no I/O. The service layer takes ports. Unit tests use fakes. Integration tests use real Qdrant and MinIO.

Version 2: the SDK is a drop-in `QdrantClient`. The old `Client` facade is removed.

## Goal of version 2

`ShardedQdrantClient` is a subclass of `QdrantClient`. A user replaces the class name and keeps all code. Hover, signatures and docstrings in the IDE are those of `QdrantClient`. For a collection that has a map, the client adds `shard_key_selector` to each call. For a collection without a map, the client does nothing.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Drop-in | Subclass `QdrantClient`. Each override is a normal method with `@functools.wraps(QdrantClient.<name>)`. `sharded_client.pyi` lists only `__init__`, `sharding`, and `ShardingAdmin`. | Pyright skips unreachable code, so `if not TYPE_CHECKING` hid `upsert` from completion. The stub makes pyright inherit Qdrant's methods and hide private helpers. `wraps` keeps `inspect.signature`, `help()`, and Jupyter on the real signature. Refresh the stub `__init__` when the qdrant-client floor or target changes. |
| Explicit selector | If the caller passes `shard_key_selector`, the call goes through unchanged. | Power users keep control. The admin code uses this path, so no recursion. |
| Which collections are sharded | A collection is sharded when its map file exists. | One source of truth. |
| Map location | `MAP_URI` is a folder: `file:///dir/`, `C:\dir\`, or `s3://bucket/prefix/`. Map of collection `c` is `<folder>/c.json`. | One folder per Qdrant instance. |
| Feature field | Any payload field, set per collection. Values are `int` or `str`. `feature_type` is `"integer"` or `"keyword"`. | The user's field, not a fixed `subject_id`. |
| Payload | A normal write stores the payload the caller sent. A point without the feature field raises `ValueError`. `alias_feature` and `drop_alias` set the feature field to the canonical id. | A filter on the new id must find the old points. Other writes stay as sent. |
| Map keys | `str(value)` in the map JSON. `MapDoc.payload_values(name)` returns typed values for filters. | JSON keys are strings. |
| Index | `"keyword"`: `KeywordIndexParams(type="keyword", is_tenant=True)`. `"integer"`: `IntegerIndexParams(type="integer", lookup=True, range=False, is_principal=True)`. | `is_tenant` exists only for keyword and uuid. |
| Move states | Write to old and new for the whole move. Reads switch from old to new in one step (see Routing). | A read always hits one placement. No dedupe. `limit`, `offset`, `count`, `facet` stay exact. |
| Grace | Wait `grace_seconds` before the copy, before the read switch, before the delete, and before an alias rewrites payloads. Default 60. Must be more than the readers' `map_ttl_seconds`. | Clients with an old cached map stay correct. |
| Read without feature in filter | Normal call without selector. Qdrant reads all shards. `FanOutWarning`. | User answer 2a and 3a. |
| Id-only operations | A custom-shard write sends a shard key. `delete`, and a filter that does not name the feature, uses every slot. `set_payload`, `delete_payload`, `delete_vectors`, and `update_vectors` by id read the stored feature, then write each point on the slots for its move state. If that feature is missing from the cached map, the client loads the map once and routes again. If it is still unknown, each id goes to the slot on its retrieved record, one key per call. The client tries a record with no shard key once on each slot. A missing id goes to the first slot. An id update never lists several shard keys in one call. During `copying` the client aligns the point onto the new slot first. During `cutover` the id update uses the new slot only. `delete` ignores a missing id. The other id writes return 404 after they update the ids that exist. `FanOutWarning`. | Qdrant 1.19 returns 400 `Shard key not specified` when the key is omitted. One write that lists several shard keys runs those keys at the same time and returns on the first error, so a 404 can arrive before the slot that holds the point is updated. One key per call avoids that. A missing id returns 404 from the first slot.  A point written in `cutover` is missing on the old slot. An upsert of an unknown feature adds the feature. An id update does not. |
| Feature value change | Safe move: upsert to new slots with `wait=True`, then delete on old slots with `wait=True`. `FeatureMovedWarning`. | Vitess and Citus do delete plus insert. A crash leaves a duplicate, never a lost point. |
| Warnings | `ShardingWarning(UserWarning)` with subclasses. `warnings.warn(..., stacklevel=3)`. Constructor flag `sharding_warnings=True`. | Pandas `PerformanceWarning`, SQLAlchemy `SAWarning`. Users can also use `warnings.filterwarnings`. |
| Admin | `client.sharding.*`. | User answer 5a. |
| Version floor | `qdrant-client>=1.12`, no upper bound. README notice: later versions can break. | `query_points` 1.10, `is_tenant` and `is_principal` 1.11, `facet` 1.12. |
| Sync, then async | `ShardedQdrantClient` first. `AsyncShardedQdrantClient` reuses the pure router. | User answer 8. |
| Hash | `zlib.crc32(str(point_id).encode())`. | Python `hash` of a `str` changes per process. |
| Stats | `client.sharding.refresh_stats(collection)` only. | A map write per upsert makes parallel loaders fight on compare-and-swap. |
| Unknown feature on write | An upsert adds the feature, then writes its slot. An id update loads the map once. If the feature is still unknown, the update uses the slot on the retrieved record.  | The owner wants new categories registered on upsert. An id update must not invent a placement. |

Kept from version 1: no `Split` class (partitions = `len(shard_keys)`), `write_to: list[str] | None`, `write_slots` and `read_slots` instead of `Route`.

## Package

```text
src/qdrant_custom_sharding/
  __init__.py          ShardedQdrantClient, warnings, errors, FeatureState
  models.py            domain: FeatureState, FeatureInfo, MapDoc (+ methods), Point
  routing.py           pick_slot, write_slots, read_slots         (pure)
  filters.py           feature_values(filter, field)                (pure)
  router.py            plan_writes, read_selector                   (pure)
  calls.py             call shape: selectors, payload rules, batch unpack, kwargs (pure)
  planning.py          plan_placement                               (pure)
  prepare.py           count_features, vector_params, collection_setup (file scan)
  map_codec.py         map_to_json, map_from_json                   (pure)
  errors.py            ShardingError, WriteDenied, VersionConflict, MapNotFound,
                       MapUnavailable, MoveNotReady, InvalidMove
  sharding_warnings.py ShardingWarning, FanOutWarning, FeatureMovedWarning,
                       StaleMapWarning, warn(category, message, enabled)
  ports.py             MapStore, MapFolder, ListableMapFolder, VectorStore Protocols
  map_stores.py        FileMapStore, S3MapStore, FileMapFolder, S3MapFolder,
                       open_map_folder(uri), register_map_folder_scheme
  qdrant_store.py      QdrantVectorStore, AsyncQdrantVectorStore (always pass a selector)
  map_cache.py         MapCache (TTL, stale fallback, negative cache, thread lock)
  stats.py             collection_stats, recommend_splits, distribution analytics (pure)
  metrics.py           MetricsRecorder, InMemoryMetrics, render_prometheus
  dashboard.py         DashboardController, JobRunner, DashboardServer, HTTP API, CLI
  dashboard_ui.py      self-contained HTML/CSS/SVG management UI
  services.py          admin service functions (take ports)
  sharded_client.py    ShardedQdrantClient, _CallRouter, ShardingAdmin
  sharded_client.pyi   public surface only: __init__, sharding, ShardingAdmin
  async_services.py    async admin steps that talk to Qdrant
  async_sharded_client.py  AsyncShardedQdrantClient, _AsyncCallRouter, AsyncShardingAdmin
  async_sharded_client.pyi public surface only: __init__, sharding, AsyncShardingAdmin
tests/
  fakes.py             FakeMapStore, FakeMapFolder, FakeVectorStore
  unit/                pure modules, map cache, services
  integration/         map stores (file, MinIO), Qdrant adapter, ShardedQdrantClient
  e2e/                 skewed collection on 3-peer Qdrant + MinIO
```

`warnings.py` is not used as a module name, so it does not hide the standard library.

## Domain (`models.py`)

```text
FeatureState(StrEnum): stable | copying | dual | cutover

FeatureInfo (dataclass)
  shard_keys: list[str]                 placement that readers use while stable/copying
  state: FeatureState = stable
  write_to: list[str] | None = None     target placement while moving
  points_count: int = 0
  aliases: list[str] = []
  grace_until: datetime | None = None   UTC, earliest time for the next move step
  copied: bool = False                  True after a successful copy_feature

MapDoc (dataclass)
  collection: str
  feature_field: str
  feature_type: str = "keyword"         "keyword" | "integer"
  slots: list[str]
  features: dict[str, FeatureInfo]      key is str(value)
  slot_load: dict[str, int]
  version: int = 1

  find(value) -> (name, FeatureInfo) | None        str(value), also matches aliases
  add_feature(value) -> FeatureInfo                least loaded slot
  start_move(feature, targets, grace_until)        stable  -> copying, copied = False
  copy_allowed(feature, now)                       copying and now >= grace_until, else MoveNotReady
  mark_copied(feature, now)                        sets copied = True, same checks
  open_dual(feature, now, grace_until)             copying -> dual, requires copied
  cutover(feature, now, grace_until) -> old slots  dual    -> cutover
  finish_move(feature, now)                        cutover -> stable, copied = False
  alias_feature(old, new)                         stable only; rename the key; append old to aliases
  drop_alias(old)                                 stable only; remove one alias
  set_stats(counts: dict[slot, dict[str, int]])
  busy_slots() -> set[str]
  least_loaded(n, exclude) -> list[str]
  payload_values(name) -> list[int | str]          name + aliases, typed by feature_type

Point(id, vector: Any, payload: dict)              vector as Qdrant returns it (list or dict)
```

Rules on `MapDoc`:

- `find` and `add_feature` take `int` or `str`. They use `str(value)`.
- `add_feature`: if `find` hits, return that row. Pick the slot with the smallest `(slot_load, features on slot)`. Skip busy slots. If every slot is busy, use all slots.
- `start_move`: only from `stable`. `targets` non-empty, all in `slots`, no repeats, none busy, none in `shard_keys`. Sets `write_to`, `grace_until`, `copied = False`. An overlap is not allowed: after a split from `s1` onto `s1, s5, s9`, the delete would keep `s1` and its copies of points that now belong on `s5` and `s9`.
- `copy_feature`: only when `state == copying` and `now >= grace_until`, else `MoveNotReady`. On success `copied = True`.
- `open_dual`: only from `copying`, else `InvalidMove`. Only when `now >= grace_until` and `copied` is true, else `MoveNotReady`.
- `cutover`, `finish_move`: only from the state before. Only when `now >= grace_until`, else `MoveNotReady`. Each sets the next `grace_until`. `finish_move` clears `grace_until` and `copied`.
- `cutover` does not change `shard_keys`. The old slots stay in `shard_keys` until `finish_move`, so the delete knows them. `cutover` returns the old slots not in `write_to`.
- `finish_move` sets `shard_keys = write_to`, `write_to = None`, `state = stable`, `grace_until = None`, `copied = False`.

- `alias_feature(old, new)`: `old` and `new` are `int` or `str`, stored with `str()`. The feature must be `stable`, else `InvalidMove`. Rename key `old` to `new`, append `old` to `aliases`. `new` must be unused.
- `drop_alias(old)`: `old` must be an alias on a `stable` feature, else `InvalidMove`. Remove `old` from `aliases`. The canonical key stays.
- The service layer bumps `version`. Domain methods do not.

## Routing (`routing.py`)

```text
pick_slot(slots, point_id) = slots[crc32(str(point_id)) % len(slots)]

old = shard_keys, new = write_to

state     write                                       read     what happens in this state
stable    [pick(old, id)]                             old      normal
copying   unique([pick(old, id), pick(new, id)])      old      wait grace, then copy old -> new
dual      unique([pick(old, id), pick(new, id)])      new      wait grace (old readers still see full old)
cutover   [pick(new, id)]                             new      wait grace, then delete old, finish
```

Why it is correct:

- `copying`: the copy starts after grace. By then every client has the `copying` map, so every new upsert goes to both. The copy inserts only missing points, so a newer write on `new` stays. A partial update copies the current point onto `new` before it changes the point. After each chunk the copy deletes ids that are no longer on `old`. The copy plus the double writes make `new` full.
- `dual`: readers switch to `new`. Clients with the old `copying` map still read `old`, and `old` is still full because writes go to both.
- `cutover`: writers stop writing `old`. Clients with the old `dual` map still write both. The delete waits for grace, so it also removes their late writes on `old`.
- A read always uses one placement. No dedupe.

## Filters (`filters.py`)

```text
feature_values(flt, field) -> list[int | str] | None
```

- `flt` is `models.Filter`, `None`, or something else (then `None`).
- Look at `flt.must` (a list or one condition). A `FieldCondition` with `key == field` and `match` of `MatchValue` gives `[value]`. `MatchAny` gives `list(any)`. A nested `Filter` in `must` is searched the same way.
- `should`, `must_not`, `min_should` are not used. They do not limit the feature.
- The first match wins. No match gives `None`.

## Router (`router.py`)

Pure functions. Input: `MapDoc` and Qdrant request objects. Output: what to call.

```text
read_selector(doc, flt) -> list[str] | None
  values = feature_values(flt, doc.feature_field)
  None                  -> None (fan-out)
  all values known      -> unique union of read_slots for each value
  any value unknown     -> None (fan-out); the client reloads the map once first,
                           because another client can have added the feature

plan_writes(doc, points) -> (dict[slot, list[PointStruct]], new_values)
  points is list[PointStruct] or Batch (Batch is turned into PointStruct list).
  value = payload[doc.feature_field]; missing field -> ValueError with point id.
  unknown values -> returned in new_values; caller adds them to the map and calls again.
  each point goes to every slot in write_slots(info, point.id).
```

## Ports (`ports.py`)

```text
MapStore
  load() -> MapDoc                       MapNotFound, MapUnavailable
  save(doc) -> None                      stored.version must be doc.version - 1
                                         (no stored doc: doc.version must be 1)
                                         else VersionConflict
  delete() -> None

MapFolder
  store_for(collection) -> MapStore

ListableMapFolder(MapFolder)
  list_collections() -> list[str]

MetricsRecorder
  record_read(collection, method, fanout, slot_count) -> None
  record_write(collection, method, fanout, slot_count) -> None
  record_feature_move(collection, count) -> None
  record_cache_event(collection, outcome) -> None
  record_write_denied(collection) -> None
  snapshot() -> RoutingMetricsSnapshot

VectorStore                              every call passes an explicit selector
  create_collection(collection, **kwargs) -> None
  create_slot(collection, slot) -> None
  create_feature_index(collection, field, feature_type) -> None
  upsert(collection, slot, points: list[Point]) -> None
  insert_missing(collection, slot, points: list[Point]) -> None   insert-only upsert
  ids_on_slot(collection, slot, ids) -> set                       retrieve those ids
  delete_ids(collection, slot, ids) -> None                       wait=True
  scroll(collection, slot, field, values, batch) -> Iterator[list[Point]]   with vectors
  delete_feature(collection, slot, field, values) -> None
  feature_counts(collection, slot, field) -> dict[str, int]
  rewrite_feature(collection, slots, field, old_values, new_value) -> int
    count exact matches of field MatchAny(old_values) on slots
    then set_payload {field: new_value}, wait=True
    return that count
```

`search` and `list_slots` leave the port. The data path is `ShardedQdrantClient` itself.

`create_collection` forwards `kwargs` (`vectors_config`, `sparse_vectors_config`, `replication_factor`, and the rest) to Qdrant. `vector_size` and `distance` are not separate arguments: they cannot describe a named or sparse vector config.

## Service layer (`services.py`)

All functions take ports. No globals. The data path is not here.

```text
update_map(store, change, retries=5) -> result
  load, deepcopy, result = change(doc), doc.version += 1, save. Retry on VersionConflict.

create_sharded_collection(store, vectors, collection, feature_field, feature_type,
                          n_slots=48, feature_counts=None, **collection_kwargs) -> MapDoc
  feature_type must be keyword or integer. Checked before any I/O.
add_feature(store, value) -> FeatureInfo
add_features(store, values) -> MapDoc
  one update_map for every value. One compare-and-swap for an upsert batch.
start_move(store, feature, targets, now, grace_seconds)
copy_feature(store, vectors, feature, now, batch=256) -> int
  requires state copying and now >= grace_until, else MoveNotReady
  scroll old slots by payload_values
  align_ids each chunk from the old slot onto pick_slot(write_to, id)
  then set copied = True
open_dual(store, feature, now, grace_seconds)
cutover(store, vectors, feature, now, grace_seconds) -> CutoverResult
  reconcile the symmetric difference, then dual -> cutover
  old_slots, copied, removed
finish_move(store, vectors, feature, now)
  delete_feature on old slots, then finish_move in the map
migrate(store, vectors, feature, to_slots, now_fn, sleep_fn, grace_seconds)
  start_move, sleep, copy_feature(now_fn()), open_dual, sleep, cutover, sleep, finish_move
split_targets(doc, feature, partitions) -> list[str]
  least loaded slots that are not busy and not in the feature's shard_keys.
  An overlap would leave duplicates after finish_move.
alias_feature(store, vectors, old, new, sleep_fn, grace_seconds) -> int
  stable only, else InvalidMove
  rename in one compare-and-swap, then sleep grace_seconds
  reload; if the feature is not stable, InvalidMove
  rewrite_feature: typed aliases -> typed canonical id, on shard_keys
  return the count
drop_alias(store, vectors, old) -> int
  old must be an alias on a stable feature, else InvalidMove
  same rewrite, then remove old from aliases
  return the count
refresh_stats(store, vectors) -> MapDoc
```

## Aliases

`alias_feature` renames the map key, then waits `grace_seconds`, then rewrites payloads.

Until the rewrite ends, a filter on the new id finds only new points.

After the rewrite, a filter on an old id still reads the same slots.

That filter finds only points written later with the old id.

Call `drop_alias` when nothing sends that old id.

`drop_alias` runs the same rewrite, then removes the alias.

After `drop_alias`, a write with the old id is an unknown feature.

The client places that write as a new feature.

Both calls require state `stable`. A running move raises `InvalidMove`.

Sync and async use the same steps. Async waits with `asyncio.sleep`.

## MapCache (`map_cache.py`)

```text
MapCache(store, ttl_seconds, now_fn)
  get() -> (MapDoc | None, Reason | None)
    fresh cache (doc or "no map")    -> cached result
    load ok                          -> (doc, None)
    MapNotFound                      -> (None, None), cached as "no map" for ttl
    MapUnavailable + cached doc      -> (cached, stale_map)
    MapUnavailable, nothing cached   -> (None, no_map)
  put(doc), forget(), expire()
    expire() clears the load time. The cached doc and the no-map flag stay.
    The next get loads again. MapUnavailable with a cached doc returns stale_map.
    forget() drops the cached map. delete_collection uses forget().
```

`Reason(StrEnum)`: `stale_map | no_map`. It moves from `models.py` to `map_cache.py`.

## ShardedQdrantClient (`sharded_client.py`)

```python
class ShardedQdrantClient(QdrantClient):
    def __init__(self, *args, map_uri: str | None = None, map_ttl_seconds: float = 30,
                 sharding_warnings: bool = True, detect_feature_changes: bool = True,
                 map_folder: MapFolder | None = None, now_fn=None, sleep_fn=None,
                 **kwargs) -> None: ...
    sharding: ShardingAdmin

    @functools.wraps(QdrantClient.upsert)
    def upsert(self, *args, **kwargs): ...
```

- `sharded_client.pyi` is the type of this class. It declares `__init__` (Qdrant 1.19 parameters, defaults as `...`, plus the keyword-only map arguments) and `sharding`. It does not list routed methods, so pyright inherits those from `QdrantClient`. Private helpers are absent from completion. Refresh that `__init__` when the qdrant-client floor or target changes.
- `@functools.wraps` copies the docstring and points `inspect.signature` at `QdrantClient`. The body only binds the call and hands it to `_CallRouter`.
- `map_uri` comes from `MAP_URI` if not given. No map folder means no collection is sharded. All calls go through unchanged.
- `map_folder` injects a folder in tests. `now_fn` and `sleep_fn` are the clock for migrate and split. `detect_feature_changes=False` skips the retrieve that looks for a changed feature.
- Each override: `bound = inspect.signature(QdrantClient.<name>).bind(self, *args, **kwargs)`, then `bound.arguments`. This handles positional arguments.
- If `shard_key_selector` is given, or the collection has no map, call `super()` unchanged.
- One `MapCache` per collection.
- Routed overrides stay on the class as normal methods. Map loads and Qdrant calls run on `self._router` (`_CallRouter`). `calls.py` chooses the selector, the ids to delete, and how a batch operation unpacks. The class surface besides `QdrantClient` is `sharding`.

| Method | Rule |
|---|---|
| `query_points`, `query_points_groups`, `count`, `scroll`, `facet`, `search_matrix_pairs`, `search_matrix_offsets` | `read_selector(doc, <filter arg>)`. `None` -> no selector + `FanOutWarning`. |
| `query_batch_points` | Same rule per request. Set `request.shard_key` on a copy. |
| `upsert` | `plan_writes`. New values go through `add_features`, then plan again. The feature change check runs before the writes. One `super().upsert(..., shard_key_selector=slot)` per slot. `wait=True` only when a point moves, so the delete is safe. |
| `upload_points`, `upload_collection` | Cut into `batch_size` chunks of `PointStruct`. Each chunk goes through the routed `upsert`. `parallel > 1` gives a `ShardingWarning` and runs with 1. Routed upload does not use `max_retries` or `method`. No map still uses the native upload. |
| `retrieve` | No selector. Qdrant reads all shards. `FanOutWarning`. |
| `update_vectors` | Read the feature for the ids. Slots follow the partial-update table. `FanOutWarning`. |
| `delete`, `set_payload`, `delete_payload`, `delete_vectors` | A filter that names the feature uses `shard_keys` plus `write_to`. `read_slots` alone misses the old copy during cutover. `delete`, a filter with no feature, or an unknown value uses every slot. An id list other than `delete` uses the partial-update table. `FanOutWarning` when the call does not name one feature. |
| `set_payload`, `overwrite_payload` that write the feature field | Safe move (below). |
| `overwrite_payload` that omits the feature field | `ValueError`. The point would have no feature. A nested `key` is not a feature move. |
| `delete_payload` with the feature field, `clear_payload` | `ValueError`: the point would have no feature. |
| `batch_update_points` | Each operation runs through the routed method for its type, in order. |
| `delete_collection` | `super()`, then delete the map. |
| Other methods | Not overridden. Legacy `search`, `recommend`, `discover` in old clients fan out in Qdrant. Results are correct. |

`delete`, `set_payload`, `delete_payload`, and `delete_vectors` with a feature filter use `shard_keys` plus `write_to`, so both copies change during a move, including cutover.

### Partial updates during a move

A partial update is `set_payload`, `overwrite_payload`, `delete_payload`, `update_vectors`, or `delete_vectors`. A payload write that sets the feature field still uses the safe move. `delete` does not use this copy.

| State | Id partial update | Filter partial update |
|---|---|---|
| `stable` | The old slot. | A named feature uses `shard_keys` and `write_to`. No feature uses every slot. |
| `copying` | Copy the point onto the new slot, then update both slots. | Scroll the matching ids on the old slots, copy those points, then send the filter as in `stable`. |
| `dual` | The old slot and the new slot. | Same as `stable`. |
| `cutover` | The new slot. | Same as `stable`. |

A retrieved point with no feature field uses the slot on the record, the same as an unknown value. The client loads the map once. The client sends a record with no shard key once on each slot. A 404 from an empty slot does not cancel the other slots. 

During `copying`, align reads the ids on the old slot. The read uses that slot and includes payload and vectors. Present ids are inserted on the new slot with insert-only. Absent ids are deleted on the new slot. Align reads the old slot once more. It repeats the insert or the delete when presence changed. The caller's update then runs on both slots for ids that are still present. Ids that are gone use the normal missing-id rule.

A filter that names a copying feature scrolls that feature's old slots. A filter with no feature does the same for every feature in `copying`. The scroll reads ids and the feature field. The client aligns those points, then sends the filter write as before.

Before cutover, the client scrolls feature ids on the old slots and on the new slots. The scroll reads ids and the feature field. Align runs for ids that are on one side only. `cutover` returns how many points were copied and how many were removed.

Insert-only needs qdrant-client and the server at 1.17 or later. An older client raises `ShardingError`. `copy_feature` raises the same error.

The bulk copy inserts only a missing point. After align, the point is on the new slot, so a later insert of an older snapshot does not run.

A delete and a write of the same point by different clients, within milliseconds, can leave the copies different until the reconcile before cutover. Two clients can write different values to the same point at the same time. The two copies can apply those writes in a different order.

### Feature change check (upsert)

1. Before the upsert: `super().retrieve(ids, with_payload=[field], with_vectors=False)` with no selector. One call per batch.
2. A point moves when the stored value and the new value are different features, or when an old write slot is not a new write slot. An alias and its canonical id are the same feature. That re-upsert is not a move and deletes nothing.
3. Upsert to the new slots. `wait=True` when any point moves; otherwise the caller's `wait` stays.
4. `super().delete(ids, shard_key_selector=old_slot, wait=True)` per old slot.
5. One `FeatureMovedWarning` per call with the count.

`retrieve` returns no shard key. Step 2 uses the old value: old slots = `write_slots(old info, id)` that are not in the new write slots.

### Safe move for `set_payload` and `overwrite_payload`

1. Find points: `retrieve(ids)` or `scroll(filter)` with vectors and payload, no selector.
2. Build the new payload.
3. Routed upsert of the new points (`wait=True`).
4. Delete on old slots (`wait=True`).
5. `FeatureMovedWarning`.

### Map outage

Stale cached map (`Reason.stale_map`): reads and writes use it, with `StaleMapWarning`. While the map store is down, the admin cannot change the map either, so the cached map can only be older by at most one TTL. The grace protocol already allows that.

An id write or a filter read that reloads the map calls `expire()`. The cached map stays. A failed load returns that map with `stale_map`.

`MapUnavailable` and nothing cached: the client does not know if the collection is sharded. It calls `get_collection(name)` once and caches `sharding_method`. `custom`: reads fan out with `StaleMapWarning`, writes raise `WriteDenied`. `auto`: pass through.

## Async client

`AsyncShardedQdrantClient` subclasses `AsyncQdrantClient`. Routed methods are normal `async def` methods with `@functools.wraps(AsyncQdrantClient.<name>)`. `async_sharded_client.pyi` lists only `__init__`, `sharding`, and `AsyncShardingAdmin`, so pyright inherits the parent methods.

| Piece | What it does |
|---|---|
| `router.py`, `calls.py` | Pure decisions. Both clients call these. |
| `sharded_client.py`, `async_sharded_client.py` | I/O only. The async methods match the sync methods, with `await`. |
| Map cache and map stores | Sync. The async client calls them with `asyncio.to_thread`. |
| `AsyncQdrantVectorStore`, `async_services.py` | Qdrant admin I/O. Map updates go through `update_map` on a worker thread. `asyncio.sleep` is the migrate wait. |

`client.sharding` is `AsyncShardingAdmin`. The method names match `ShardingAdmin`. Refresh the stub `__init__` when the qdrant-client floor or target changes.

## ShardingAdmin (`client.sharding`)

```text
create_collection(collection_name, feature_field, feature_type="keyword", n_slots=48,
                  feature_counts=None, **create_collection_kwargs) -> MapDoc
add_feature(collection, value)
migrate(collection, value, to_slot, grace_seconds=60)
split(collection, value, partitions, grace_seconds=60)
start_move / copy_feature / open_dual / cutover / finish_move(collection, value, ...)
alias_feature(collection, old, new, grace_seconds=60) -> int
drop_alias(collection, old) -> int
refresh_stats(collection) -> MapDoc
stats(collection, live=False) -> CollectionStats
list_collections() -> list[str]
get_map(collection) -> MapDoc | None
metrics: MetricsRecorder (property)
```

## Stats, metrics, and dashboard (`stats.py`, `metrics.py`, `dashboard.py`)

- `stats.py` is pure and does no I/O. `collection_stats(doc, snapshot=None, now=None)` computes per-slot loads, mean/median/min/max/p95/std, imbalance ratio, Gini coefficient, feature size histogram buckets, active move grace timers, and `recommend_splits(...)` for oversized features.
- `metrics.py` defines `MetricsRecorder`, thread-safe `InMemoryMetrics`, and `render_prometheus(collections, routing=None, top_features=50)` in Prometheus text exposition 0.0.4 with zero external dependencies.
- `dashboard.py` separates the framework-agnostic `DashboardController` and background `JobRunner` from the stdlib `ThreadingHTTPServer` (`DashboardServer`, `create_dashboard_server`, `serve_dashboard`) and CLI entry point `qdrant-sharding-dashboard`.
- `dashboard_ui.py` holds the self-contained HTML/CSS/SVG UI served at `/`.
- `map_stores.py` exposes `register_map_folder_scheme` and `unregister_map_folder_scheme` so callers can attach custom `MapFolder` implementations to `open_map_folder(uri)`.

`create_collection` passes `sharding_method=CUSTOM` and `shard_number=1` to Qdrant. It does not let the caller change them.

`feature_counts` keys are strings. `prepare.count_features` does that conversion. `prepare.vector_params` reads the vector length and takes the distance the embeddings were built with. `prepare.collection_setup(path, feature_field, vector_field, distance)` scans a `.csv`, `.json`, `.jsonl`, `.npz`, or `.parquet` file once and returns both arguments. A CSV vector column is a JSON list, or `vector_field` is a list of column names. Parquet needs `pyarrow`, which is not a dependency of the package. Arrays already in memory, including a memmap, use `count_features` and `vector_params` directly.

## Map stores (`map_stores.py`)

- `FileMapStore(path)`: lock file `path + ".lock"` with `os.O_CREAT | os.O_EXCL`, short retry loop, lock older than 30 s is removed. Write temp file, `os.replace`. `OSError` on read becomes `MapUnavailable`.
- `S3MapStore(bucket, key, client=None)`: `save` reads `ETag`, then `put_object(IfMatch=etag)`, or `IfNoneMatch="*"` for version 1. HTTP 412 or 409 becomes `VersionConflict`. `NoSuchKey` becomes `MapNotFound`. Other `ClientError` and connection errors become `MapUnavailable`.
- boto3 client: `endpoint_url` from `AWS_ENDPOINT_URL`, region `us-east-1` if none is set, `Config(signature_version="s3v4", s3={"addressing_style": "path"}, request_checksum_calculation="when_required", response_checksum_validation="when_required", connect_timeout=2, read_timeout=5, retries={"total_max_attempts": 2, "mode": "standard"})`. A cold load with S3 down fails in about 5 s, not minutes. Network errors are `botocore.exceptions.ConnectionError` and `HTTPClientError`; both become `MapUnavailable`. Missing credentials are a setup error and are not caught.
- `open_map_folder(uri)`: `s3://bucket/prefix/` -> `S3MapFolder`. `file:///C:/dir/`, `file:///dir/` or a plain path -> `FileMapFolder`. `store_for(c)` gives the store for `<folder>/<c>.json`.

## Qdrant adapter (`qdrant_store.py`)

Wraps the same `ShardedQdrantClient`. Every call passes an explicit `shard_key_selector`, so routing does not run again.

- `create_collection(collection, **kwargs)`. `sharding_method=CUSTOM` and `shard_number=1` are set only when the caller omits them. Admin always sets both and rejects a caller value.
- `create_shard_key(collection, slot)`.
- Index by `feature_type` (see Decisions).
- `scroll` with `shard_key_selector=slot`, `with_vectors=True`, `MatchAny(any=values)`.
- `insert_missing`: `upsert(..., update_mode=UpdateMode.INSERT_ONLY, wait=True)`. No `UpdateMode.INSERT_ONLY` raises `ShardingError`: moves need qdrant-client and server >= 1.17 for insert-only upserts. Routing stays available on 1.12.
- `ids_on_slot`: `retrieve(ids, shard_key_selector=slot, with_payload=False, with_vectors=False)`.
- `read_ids`: the same retrieve with payload and vectors. `scroll_ids`: ids and the feature field only.
- `delete_ids`: `delete(ids, shard_key_selector=slot, wait=True)`.
- `delete_feature`: `delete(FilterSelector(MatchAny(values)), shard_key_selector=slot, wait=True)`.
- `rewrite_feature`: `count` with `exact=True` on `shard_key_selector=slots`, then `set_payload` with `wait=True`. The filter is `MatchAny(old_values)`.
- `feature_counts`: `facet(key=field, shard_key_selector=slot, exact=True, limit=100_000)`, keys as `str`.
- Points keep any vector shape (unnamed, named, sparse).

## Local stack

`docker-compose.yml`: three Qdrant peers (`--uri` on node 1, `--bootstrap` on nodes 2 and 3), REST ports 6333, 6433, 6533. MinIO on 9000, image from `MINIO_IMAGE`, bucket `sharding-maps`.

## Tests

- pytest markers: `integration`, `e2e`. Default run is unit only.
- Unit: pure modules, `MapCache`, services with fakes.
- Integration: map stores on a temp folder and MinIO. `QdrantVectorStore` and `ShardedQdrantClient` on the 3-peer cluster: each routed method, warnings, safe move, positional arguments, explicit selector pass-through, unsharded collection pass-through.
- e2e: Zipf-like skewed dataset (int feature and str feature), MinIO map folder. Checks: plain `QdrantClient` code runs unchanged; each filtered query reads only its slots; a moved feature keeps all points during and after the move; a split feature spreads across slots; an alias finds the same points; a read without a feature fans out with a warning.
- `make test-min-client`: unit and integration with `qdrant-client==1.12.*`. Move tests skip. That client has no insert-only upsert.
