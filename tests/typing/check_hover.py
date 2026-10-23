from qdrant_client import AsyncQdrantClient, QdrantClient

from qdrant_custom_sharding import AsyncShardedQdrantClient, ShardedQdrantClient

sync_client = ShardedQdrantClient("http://localhost:6333")
async_client = AsyncShardedQdrantClient("http://localhost:6333")

reveal_type(QdrantClient.upsert)
reveal_type(sync_client.upsert)
reveal_type(QdrantClient.query_points)
reveal_type(sync_client.query_points)
reveal_type(sync_client.sharding)
reveal_type(ShardedQdrantClient.__init__)
reveal_type(AsyncQdrantClient.upsert)
reveal_type(async_client.upsert)
reveal_type(AsyncQdrantClient.query_points)
reveal_type(async_client.query_points)
reveal_type(async_client.sharding)
reveal_type(AsyncShardedQdrantClient.__init__)
