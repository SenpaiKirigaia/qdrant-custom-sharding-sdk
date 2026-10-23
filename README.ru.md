# qdrant-custom-sharding-sdk

English: [README.md](README.md).

Пакет ставится как `qdrant-custom-sharding-sdk`. Импорт: `qdrant_custom_sharding`.

## Что это

`ShardedQdrantClient` — подкласс `QdrantClient`. Меняете имя класса, передаёте папку карты и оставляете остальные вызовы как есть. Если у коллекции есть карта, клиент сам подставляет `shard_key_selector` по признаку из фильтра или payload. Коллекция без карты идёт в Qdrant без изменений.

Признак — любое поле payload, своё для каждой коллекции, например `subject_id`. Значения — `int` или `str`. Клиент укладывает тысячи таких значений не больше чем в 48 слотов. Слот — это один ключ шарда Qdrant. Карта, которая говорит, какое значение на каком слоте лежит, — один JSON на коллекцию, в локальной папке или под префиксом S3 или MinIO.

## Зачем

Обычная коллекция Qdrant держит каждую точку на каждом шарде. Поиск, подсчёт и обновление ждут ответа от всех шардов, даже если нужен один `subject_id`. Если почти каждый вызов остаётся внутри одного признака, это ожидание и есть потеря скорости.

Custom sharding умеет отправить вызов на один ключ шарда, но ключей Qdrant хочет мало. Один ключ на каждый `subject_id` не вмещает тысячи id, а несколько крупных признаков всё равно сели бы на один шард и перегрузили его. Этот SDK хранит карту: значение признака указывает на один или несколько слотов, не больше 48. Много мелких значений делят слот. Крупное значение может занять несколько слотов, и клиент выбирает слот по `crc32` от id точки.

`shard_key_selector` передавать не нужно. Хватает фильтра `must` по полю признака, того же фильтра, которым вы и так ищете внутри одного субъекта. Клиент читает значение, смотрит карту и добавляет слот. Фильтр без признака по-прежнему читает все шарды, и клиент шлёт `FanOutWarning`: он не знает, какого слота достаточно.

## Установка

Нужны Python 3.11 или новее, `qdrant-client` 1.12 или новее и сервер Qdrant 1.12 или новее. Пакет также ставит `boto3` 1.36 или новее. Верхней границы у `qdrant-client` нет: SDK перекрывает публичные методы `QdrantClient`, и более новый релиз может убрать или изменить метод. Тогда этого метода нет и здесь. В проде зафиксируйте проверенную версию. Проверенные клиенты — 1.19.1 и 1.12.2. Проверенный сервер — 1.19.2.

Пакета нет на PyPI. Ставьте его из локальной копии или из git, если у репозитория есть remote.

```text
uv add path/to/qdrant-custom-sharding-sdk
pip install path/to/qdrant-custom-sharding-sdk
uv add git+<url>
pip install git+<url>
```

В этом репозитории `uv sync` и `make install` делают одно и то же.

## Быстрый старт

Замените `QdrantClient` на `ShardedQdrantClient`, передайте `url` и `map_uri` и создайте коллекцию через `client.sharding.create_collection`. После этого `upsert`, `query_points`, `count` и `scroll` остаются обычными вызовами Qdrant. Положите поле признака в фильтр, чтобы клиент выбрал слот.

Обычный `create_collection` карту не пишет, и клиент такую коллекцию не маршрутизирует. `client.sharding.create_collection` создаёт коллекцию, слоты, индекс payload и карту. Имена слотов идут от `s0` вверх, одно имя на слот, до `n_slots`. Ключи в `feature_counts` — строки, поэтому целое 7 — это ключ `"7"`. Для `feature_type="keyword"` индекс тенантный. Для `feature_type="integer"` индекс lookup, range выключен.

Точка без поля признака даёт `ValueError`. Upsert нового значения сначала добавляет его в карту, потом пишет точки.

Считать признаки руками не нужно. `collection_setup` читает CSV, JSON, JSONL, NumPy `.npz` или Parquet и возвращает `feature_counts` и `vectors_config`. Расстояние задаёте вы: файл не знает, векторы считались для cosine или для dot. Для Parquet нужен `pyarrow`. Массив NumPy или memmap, который уже в памяти, идёт через `count_features` и `vector_params`. Каждый формат разобран в блокноте [examples/new_collection.ipynb](examples/new_collection.ipynb).

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

В CSV колонка `vector` — это JSON-список. Для широкого CSV вместо имени колонки передайте список: `["v0", "v1", "v2", "v3"]`.

### Признак int

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

### Признак str

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

## Асинхронный клиент

`AsyncShardedQdrantClient` — то же самое для `AsyncQdrantClient`. Аргументы конструктора совпадают с родителем. Админские вызовы — `await client.sharding.<method>(...)`. Хранилища карты синхронные, и boto3 или локальный файл заблокировали бы цикл событий, поэтому асинхронный клиент читает карту в рабочем потоке через `asyncio.to_thread`.

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

## Поддержка редактора

Дополнение, подсказка и сигнатуры берутся у `QdrantClient` или `AsyncQdrantClient` той версии `qdrant-client`, которая установлена. В пакете лежат заглушки `.pyi` и маркер `py.typed`. Маршрутизируемые методы в заглушке не повторяются, поэтому проверщик типов наследует их у родителя, а `help(client.upsert)` показывает сигнатуру Qdrant: каждый override обёрнут в `functools.wraps`. После установки или обновления перезапустите языковой сервер.

Заглушка копирует список параметров конструктора из `qdrant-client` 1.19.1. На другой версии клиента подсказка конструктора может отличаться от конструктора, который вы вызываете.

## Настройка

| Аргумент | По умолчанию | Эффект |
|---|---|---|
| `map_uri` | переменная `MAP_URI` или ничего | папка файлов карты |
| `map_ttl_seconds` | 30 | секунды до следующей загрузки |
| `sharding_warnings` | `True` | слать предупреждения |
| `detect_feature_changes` | `True` | читать старое значение перед upsert |

`map_folder`, `now_fn` и `sleep_fn` нужны тестам: туда подставляют папку и часы. Без `map_uri` и без `MAP_URI` ничего не шардируется, а админский вызов даёт `ValueError` с текстом `no map folder`.

### Папка карты

URI — это папка, не один файл. Каждая коллекция — `<folder>/<collection>.json`. Одна папка на один экземпляр Qdrant.

- Путь к папке: `C:/qdrant-maps` или `C:\qdrant-maps`
- Файловый URI: `file:///C:/qdrant-maps/`
- S3 или MinIO: `s3://bucket/prefix/`

### S3

boto3 читает ключи из окружения. Не передавайте их в клиент.

| Переменная | Роль |
|---|---|
| `AWS_ENDPOINT_URL` | свой endpoint, например MinIO |
| `AWS_ACCESS_KEY_ID` | ключ доступа |
| `AWS_SECRET_ACCESS_KEY` | секретный ключ |
| `AWS_DEFAULT_REGION` | регион |

SDK берёт регион из `AWS_DEFAULT_REGION` или из `AWS_REGION`. Свой endpoint без региона получает `us-east-1`. Без `AWS_ENDPOINT_URL` boto3 использует обычные настройки AWS.

`boto3` 1.36 по умолчанию считает контрольные суммы, и часть S3-совместимых хранилищ из-за этого ломается, поэтому SDK ставит `request_checksum_calculation` и `response_checksum_validation` в `when_required`. Запись карты условная: первая шлёт `If-None-Match: *`, следующая — `If-Match` с текущим ETag. AWS S3 и MinIO эти заголовки принимают. Нет ключа доступа или секрета — это ошибка настройки. Её бросает botocore, SDK её не перехватывает.

PowerShell для локального MinIO:

```powershell
$env:MAP_URI = "s3://sharding-maps/app/"
$env:AWS_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin123"
$env:AWS_DEFAULT_REGION = "us-east-1"
```

## Как устроена маршрутизация

Если вы сами передали `shard_key_selector`, вызов уходит в Qdrant как есть. То же самое, если у коллекции нет карты. Иначе клиент выбирает слоты по признаку.

Признак выбирает только условие `must`, в том числе `Filter`, вложенный в `must`. Клиент читает `MatchValue` и `MatchAny`. Диапазон не считается, как и `should`, `must_not`, `min_should` и `NestedCondition`. Такие вызовы идут на все шарды.

Чтение:

| Метод | Правило |
|---|---|
| `query_points` | по `query_filter` |
| `query_points_groups` | по `query_filter` |
| `count` | по `count_filter` |
| `scroll` | по `scroll_filter` |
| `facet` | по `facet_filter` |
| `search_matrix_pairs` | по `query_filter` |
| `search_matrix_offsets` | по `query_filter` |
| `query_batch_points` | по фильтру каждого запроса |

`query_batch_points` ставит `shard_key` на копию каждого запроса. Запрос, у которого `shard_key` уже есть, не меняется.

Запись:

| Метод | Правило |
|---|---|
| `upsert` | по признаку в payload |
| `upload_points` | через `upsert` |
| `upload_collection` | через `upsert` |
| `delete` | фильтр по признаку: обе копии. Иначе все слоты |
| `set_payload` | фильтр по признаку или безопасный перенос. Без признака: все слоты или слот точки для id |
| `overwrite_payload` | безопасный перенос, если в payload есть признак |
| `delete_payload` | фильтр по признаку: обе копии. Без признака: все слоты или слот точки для id |
| `delete_vectors` | фильтр по признаку: обе копии. Без признака: все слоты или слот точки для id |
| `batch_update_points` | каждая операция по порядку |

Прочие вызовы:

| Метод | Правило |
|---|---|
| `retrieve` | без ключа шарда, `FanOutWarning` |
| `update_vectors` | слот, где лежит каждый id, `FanOutWarning` |
| `delete_collection` | затем удаление карты |

Upsert неизвестного признака добавляет его и пишет на его слот. Обновление по id один раз перечитывает карту. Если признак всё ещё неизвестен или у точки нет поля признака, запись идёт на слот из ответа `retrieve`, по одному ключу шарда на вызов. Отсутствующий id идёт на первый слот и даёт 404. Если у записи нет `shard_key`, клиент пробует слоты по одному, и 404 с пустого слота не отменяет остальные.

Эти вызовы шлют `FanOutWarning`, потому что не называют один признак:

- у `retrieve` нет ключа шарда, Qdrant читает все шарды
- `update_vectors` берёт слот, где лежит каждый id
- `delete` по id или по фильтру без признака идёт на все слоты
- `set_payload`, `delete_payload` и `delete_vectors` по id идут на слот точки
- те же три по фильтру без признака идут на все слоты
- `NestedCondition`, `should`, `must_not`, `min_should` и неизвестное значение после одной перезагрузки карты

Если карта есть и `shard_key_selector` не передан, `clear_payload` даёт `ValueError`. То же делает `delete_payload`, когда в `keys` есть поле признака, и `overwrite_payload`, когда в payload нет поля признака. У точки не осталось бы признака.

`delete`, `set_payload`, `delete_payload` и `delete_vectors` пишут и в `shard_keys`, и в `write_to`, когда фильтр называет признак, поэтому во время переноса меняются обе копии, включая `cutover`. Частичное обновление — это `set_payload`, `overwrite_payload`, `delete_payload`, `update_vectors` и `delete_vectors`. `delete` точку заранее не копирует.

| Состояние | Запись по id |
|---|---|
| `stable` | старый слот |
| `copying` | клиент копирует точку на новый слот, затем меняет оба слота |
| `dual` | оба слота |
| `cutover` | новый слот |

В состоянии `copying` фильтр, который называет этот признак, сначала копирует подходящие точки. Фильтр без признака делает то же, если какой-то признак в `copying`. Затем запись по фильтру идёт на те же слоты, что и раньше.

Два клиента могут одновременно записать разные значения в одну точку, и две копии могут применить эти записи в разном порядке. SDK не перекрывает `search`, `recommend` и `discover`. Эти методы есть только у старых клиентов. Qdrant шлёт их на все шарды, результаты остаются верными.

### Предупреждения

`ShardingWarning` наследует `UserWarning`. `FanOutWarning` значит, что вызов не назвал один признак и клиент делает лишнюю работу. `FeatureMovedWarning` значит, что точки переехали на новое значение признака. `StaleMapWarning` значит, что клиент оставил кэшированную карту или у custom-коллекции карты не было.

Передайте `sharding_warnings=False`, чтобы их выключить, или игнорируйте категорию через `warnings.filterwarnings`. `ShardingWarning` глушит все предупреждения SDK. Один подкласс глушит только его.

```python
import warnings

from qdrant_custom_sharding import ShardingWarning

warnings.filterwarnings("ignore", category=ShardingWarning)
```

### Смена значения признака

`upsert`, `set_payload` и `overwrite_payload` могут сменить признак у точки. SDK пишет точку на новые слоты с `wait=True`, затем удаляет её на старых слотах с `wait=True`. Сбой может оставить вторую копию, но первая остаётся, поэтому точка не теряется. Клиент шлёт одно `FeatureMovedWarning` с числом перенесённых точек.

На upsert клиент читает старый payload, если `detect_feature_changes` истинно. Поставьте `False`, чтобы пропустить это чтение. `set_payload` и `overwrite_payload` флаг не смотрят. Вызов `set_payload` с заданным `key` точку не переносит.

## Администрирование

Операции с картой живут на `client.sharding`. Ошибки SDK наследуют `ShardingError`. Две админские записи могут конфликтовать. SDK повторяет сохранение пять раз, затем бросает `VersionConflict`. Upsert счётчики в карте не обновляет. Вызовите `refresh_stats`, когда нужны числа из Qdrant.

Не передавайте `sharding_method` и `shard_number`. SDK сам ставит custom sharding и `shard_number` 1. Остальные аргументы идут в `create_collection` Qdrant, например `vectors_config`. `feature_type` — это `keyword` или `integer`.

- `create_collection(collection_name, feature_field, feature_type="keyword", n_slots=48, feature_counts=None, **kwargs)` создаёт коллекцию, слоты, индекс и карту.
- `add_feature(collection, value)` кладёт значение на наименее загруженный слот.
- `migrate(collection, value, to_slot, grace_seconds=60)` переносит признак на один слот и ждёт `grace_seconds`.
- `split(collection, value, partitions, grace_seconds=60)` раскладывает признак на `partitions` свободных слотов.
- `start_move(collection, value, targets, grace_seconds=60)` меняет `stable` на `copying`.
- `copy_feature(collection, value, batch=256)` копирует точки на `write_to` и возвращает число точек.
- `open_dual(collection, value, grace_seconds=60)` меняет `copying` на `dual`.
- `cutover(collection, value, grace_seconds=60)` меняет `dual` на `cutover`. Возвращает `CutoverResult` с `old_slots`, `copied` и `removed`.
- `finish_move(collection, value)` удаляет признак на старых слотах, затем меняет `cutover` на `stable`.
- `alias_feature(collection, old, new, grace_seconds=60)` переименовывает ключ карты. См. «Псевдонимы».
- `drop_alias(collection, old)` переписывает точки со старым id, затем удаляет этот псевдоним.
- `refresh_stats(collection)` пишет `points_count` и `slot_load` из Qdrant.
- `get_map(collection)` возвращает карту или `None`, если файла карты нет.

### Состояния переноса

Клиент выбирает слот по `crc32` от `str(point_id)`. `shard_keys` — куда смотрят читатели в состояниях `stable` и `copying`. `write_to` — целевой список во время переноса.

| Состояние | Слоты записи | Слоты чтения |
|---|---|---|
| `stable` | один слот из `shard_keys` | `shard_keys` |
| `copying` | старый слот и новый слот | `shard_keys` |
| `dual` | старый слот и новый слот | `write_to` |
| `cutover` | один слот из `write_to` | `write_to` |

В `copying` и `dual` оба выбора могут назвать один и тот же слот. Чтение бьёт в одно размещение, поэтому `count`, `limit` и `offset` остаются точными. В `copying` и `dual` каждый upsert пишет и на старый, и на новый слот, чтобы точка, пришедшая в середине переноса, не осталась только на старом.

`grace_seconds` должен быть больше `map_ttl_seconds` у каждого читателя. По умолчанию это 60 и 30. Читатель со старой картой тогда попадает в размещение, которое ещё полное. Ждите `grace_until` перед `open_dual`, `cutover` и `finish_move`. Ранний вызов даёт `MoveNotReady`. Неверное состояние даёт `InvalidMove`.

Каждая цель уже должна быть слотом на карте. Цели не пересекаются с `shard_keys`, не повторяются и не заняты. Занятый слот уже участвует в переносе.

### Перенос и разбиение

`migrate` спит `grace_seconds` три раза: перед копией, перед сменой чтения и перед удалением. Вызов с `grace_seconds` 60 поэтому ждёт около трёх минут, и другие клиенты могут делать upsert во время ожиданий. Возьмите целевой слот, который признак ещё не занимает.

```python
doc = client.sharding.get_map("people")
current = doc.features["7"].shard_keys
target = next(slot for slot in doc.slots if slot not in current)
client.sharding.migrate("people", 7, target, grace_seconds=60)
```

`split` выбирает слоты сам. Свободный слот не занят и не входит в текущие `shard_keys`. Если свободных слотов меньше, чем `partitions`, `split` даёт `InvalidMove`. Ожидания те же, что у `migrate`. Для строкового признака передайте str, например `"books"`.

```python
client.sharding.split("people", 8, 3, grace_seconds=60)
doc = client.sharding.refresh_stats("people")
print(doc.slot_load)
print(doc.features["8"].points_count)
print(doc.features["8"].shard_keys)
```

### Псевдонимы

`alias_feature(collection, old, new, grace_seconds=60)` переименовывает ключ карты с `old` на `new` и оставляет `old` в `aliases`. Вызов ждёт `grace_seconds`. Это время должно быть больше `map_ttl_seconds` у каждого читателя. Затем он ставит поле признака у сохранённых точек в `new` и возвращает, сколько точек изменилось.

Пока перезапись не кончилась, фильтр по `new` находит только точки, записанные после переименования. После перезаписи фильтр по `old` находит только точки, записанные позже со значением `old`. Эти точки остаются на тех же слотах.

`drop_alias(collection, old)` делает ту же перезапись для точек, которые ещё несут `old`, затем убирает `old` из `aliases` и возвращает число. После этого запись со значением `old` — новый признак. Оба вызова требуют состояние `stable`. Идущий перенос даёт `InvalidMove`.

## Поведение при недоступной карте

Если загрузка не удалась и кэш есть, клиент берёт эту карту и шлёт `StaleMapWarning`. Если кэша нет, он один раз вызывает `get_collection` и запоминает способ шардирования. У custom-коллекции чтение тогда идёт на все шарды, тоже со `StaleMapWarning`, а запись бросает `WriteDenied`. К записям здесь относятся `upsert`, `delete`, `set_payload` и `update_vectors`. `retrieve` — чтение, поэтому при отказе он тоже идёт на все шарды. Auto-коллекция уходит без изменений: эта карта её не шардировала.

Таймаут соединения S3 — 2 секунды, таймаут чтения — 5 секунд, попыток не больше двух, поэтому холодная загрузка при лежащем S3 падает примерно за 5 секунд. `delete_collection` сначала удаляет коллекцию в Qdrant. На S3 неудачное удаление карты даёт `MapUnavailable`.

## Ограничения

`upload_points` и `upload_collection` с `parallel` больше 1 идут пачками по одной и шлют `ShardingWarning`. Этот путь не смотрит `max_retries` и `method`. Коллекция без карты по-прежнему использует родную загрузку Qdrant.

`search` и `recommend` не маршрутизируются. Они есть только в старых версиях `qdrant-client`. Qdrant шлёт их на все шарды, результаты остаются верными.

Фильтр с `NestedCondition` идёт на все шарды. Признак выбирают только условия `must`, и только `MatchValue` и `MatchAny`. Диапазон не выбирает. У коллекции одно поле признака.

Рекомендуемый максимум — 48 слотов, это же значение по умолчанию у `n_slots`. Большее число SDK не отвергает. Перенос требует `qdrant-client` и сервер Qdrant 1.17 или новее: insert-only upsert появился там. Более старый клиент на копии даёт `ShardingError`.

Удаление и запись одной точки разными клиентами в пределах миллисекунд во время переноса могут оставить копии разными до сверки перед cutover. Два клиента также могут одновременно записать разные значения в одну точку, и две копии могут применить эти записи в разном порядке.

## Локальная разработка

Поставьте Docker Desktop, uv и GNU make.

Из корня репозитория `make up` поднимает три узла Qdrant и MinIO. Узлы — один кластер. Ходите на порт 6333. REST-порты: 6333, 6433 и 6533. Порты gRPC: 6334, 6434 и 6534. Образ Qdrant — `qdrant/qdrant:v1.19.2`.

MinIO слушает порт 9000, консоль — 9001. Бакет — `sharding-maps`. Пользователь — `minioadmin`, пароль — `minioadmin123`. Образ по умолчанию берётся с quay.io. Чтобы ходить в этот MinIO из кода, задайте переменные из раздела «Настройка». Интеграционные тесты сами ставят `AWS_ENDPOINT_URL` и ключи MinIO.

| Цель | Действие |
|---|---|
| `install` | синхронизировать зависимости для разработки через uv |
| `test` | только модульные тесты |
| `test-integration` | маркер pytest integration |
| `test-e2e` | маркер pytest e2e |
| `test-all` | модульные, интеграционные и e2e одним запуском |
| `test-min-client` | модульные и интеграционные на qdrant-client 1.12 |
| `typecheck` | mypy и pyright по сигнатурам методов клиента |
| `up` | поднять стек docker compose |
| `down` | остановить стек, тома оставить |
| `reset` | остановить стек и удалить тома |
| `status` | docker compose ps |
| `logs` | последние 100 строк логов |

`make test` гоняет модульные тесты на подделках и Docker не нужен. `make test-integration` ходит в живые Qdrant и MinIO, и в синхронный, и в асинхронный клиент. `make test-e2e` грузит перекошенную коллекцию int и коллекцию str. `make test-min-client` гоняет модульные и интеграционные тесты на `qdrant-client` 1.12 и пропускает e2e. `make typecheck` запускает mypy и pyright на сигнатурах методов клиента.

Если quay.io отклоняет pull, задайте `MINIO_IMAGE` другим образом MinIO. Подходит уже скачанный тег `minio/minio`. Затем снова выполните `make up`.

Локальный HTTP-прокси в Windows рвёт соединения с localhost. Задайте `NO_PROXY` значение `localhost,127.0.0.1` или передайте в конструктор `trust_env=False`.

Если Docker пишет `Internal Server Error for API route`, остановите Docker Desktop, в PowerShell выполните `wsl --shutdown` и снова запустите Docker Desktop.

## Заметки по устройству

В коде порты, адаптеры и слой сервисов, как в книге Percival и Gregory [Architecture Patterns with Python](https://www.cosmicpython.com/).

Перенос пишет в оба размещения, переключает чтение, затем удаляет старые точки. Тот же порядок у [Vitess MoveTables](https://vitess.io/docs/reference/vreplication/movetables/) и у переноса слотов [Redis Cluster](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/), поэтому сбой оставляет дубликат, а не потерянную точку.

Запись карты сверяет номер версии и пишет только при совпадении. Несовпадение даёт `VersionConflict`. В S3 это `If-Match`, а для первой записи `If-None-Match`. Так же работает поле версии в обновлении MongoDB. См. [атомарность записи](https://www.mongodb.com/docs/manual/core/write-operations-atomicity/).

Раскладка берёт сначала самые крупные признаки. Это longest-processing-time-first. Описание: [longest-processing-time-first scheduling](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling). Порог — сумма точек, делённая на число слотов, с округлением вверх. Признак крупнее порога занимает несколько слотов, и его точки расходятся по `crc32` от id точки.

## Исходные цели проекта

Текст ниже скопирован без правок.

Что мы хотим уметь делать с помощью данного сдк:

1. Создавать коллекцию с кастомным шардированием или хранением векторов в конкретных шардах по некоему признаку в payload вектора.
2. Упрощение клиентской части с помощью ввода посреднкиа (S3/local) для хранения актуальной map между шардами.
3. Возможность в online режиме ребалансировать шарды. В режиме онлайн добавлять новые уникальные признаки в map.
4. Возможность поменять номер признака в рамках одного признака. (был признак а id 1, стал признак а id 239750945).
5. Иметь gui на клиентской части для того чтобы можно было отслеживать состояние этой шардированной коллекции, какой признак к какому шарду относится, как распределены вектора в рамках конкретного признака и так далее
6. Иметь возможность делить большие принаки на несколько шардов, чтобы делать запросы только туда а не на все шарды сразу

Для чего это делается?
При дефолтной загузке коллекции в квадрант мы получаем все вектора на всех шардах. При любой операции это заставляет нас делать fan-out на все шарды, ждать ответа от всех шардов, обрабатывать эти ответы, что заставляет нас терять производительность. Следовательно предложение, в зависимости от задачи можно разбить данные на разные части, например, категории. И если все операции проводятся в рамках одной категории то можно разложить их отдельно друг от друга по шардам и делать всё в рамках конкретного шарда.
