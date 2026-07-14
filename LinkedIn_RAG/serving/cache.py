"""Redis caching layer for query results."""

import json
import hashlib
import redis


class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, ttl: int = 3600):
        self.ttl = ttl
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            self.client.ping()
            print(f"Redis connected: {self.client.connection_pool.connection_kwargs['host']}:"
                  f"{self.client.connection_pool.connection_kwargs['port']}")
        except redis.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Redis. Start Redis server first:\n"
                "  Docker: docker run -d -p 6379:6379 redis:alpine\n"
                "  Windows: https://github.com/microsoftarchive/redis/releases"
            )

    def _make_key(self, query: str, top_k: int, enable_guard: bool) -> str:
        """Create a deterministic cache key from query parameters."""
        raw = f"{query.strip().lower()}:{top_k}:{enable_guard}"
        return f"talentrag:{hashlib.md5(raw.encode()).hexdigest()}"

    def get(self, query: str, top_k: int = 5, enable_guard: bool = True) -> dict | None:
        """Retrieve cached result, or None if miss."""
        key = self._make_key(query, top_k, enable_guard)
        cached = self.client.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, query: str, top_k: int, enable_guard: bool, result: dict) -> None:
        """Cache a pipeline result."""
        key = self._make_key(query, top_k, enable_guard)
        # Store only serializable fields
        cache_data = {
            "query": result.get("query", query),
            "answer": result.get("answer", ""),
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "chunk_type": c.get("chunk_type", ""),
                    "metadata": c.get("metadata", {}),
                    "rerank_score": c.get("rerank_score", c.get("rrf_score", 0.0)),
                }
                for c in result.get("chunks", [])
            ],
            "guard": result.get("guard", {}),
            "cached": True,
        }
        self.client.setex(key, self.ttl, json.dumps(cache_data))

    def invalidate_all(self) -> int:
        """Clear all TalentRAG cache entries."""
        keys = list(self.client.scan_iter(match="talentrag:*"))
        if keys:
            return self.client.delete(*keys)
        return 0

    def stats(self) -> dict:
        """Return cache stats."""
        keys = list(self.client.scan_iter(match="talentrag:*"))
        info = self.client.info("stats")
        return {
            "cached_queries": len(keys),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
        }
