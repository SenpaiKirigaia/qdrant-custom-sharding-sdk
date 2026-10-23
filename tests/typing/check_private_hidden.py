from qdrant_custom_sharding import AsyncShardedQdrantClient, ShardedQdrantClient

sync_client = ShardedQdrantClient("http://localhost:6333")
async_client = AsyncShardedQdrantClient("http://localhost:6333")

sync_client._upsert
async_client._upsert
