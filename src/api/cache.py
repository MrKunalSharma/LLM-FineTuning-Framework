"""Caching layer for inference optimization."""
import hashlib
import json
from typing import Optional, Dict, Any
import redis
from datetime import timedelta

class InferenceCache:
    """Redis-based caching for inference results."""
    
    def __init__(
        self, 
        redis_host: str = "localhost",
        redis_port: int = 6379,
        ttl: int = 3600  # 1 hour default TTL
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.ttl = ttl
    
    def _generate_key(self, instruction: str, input_text: str, params: Dict) -> str:
        """Generate cache key from request."""
        content = f"{instruction}:{input_text}:{json.dumps(params, sort_keys=True)}"
        return f"inference:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get(self, instruction: str, input_text: str, params: Dict) -> Optional[str]:
        """Get cached result."""
        key = self._generate_key(instruction, input_text, params)
        result = self.redis_client.get(key)
        if result:
            logger.info(f"Cache hit for key: {key}")
            return json.loads(result)
        return None
    
    def set(
        self, 
        instruction: str, 
        input_text: str, 
        params: Dict,
        response: str,
        metadata: Dict[str, Any]
    ):
        """Cache inference result."""
        key = self._generate_key(instruction, input_text, params)
        value = {
            "response": response,
            "metadata": metadata,
            "cached_at": datetime.utcnow().isoformat()
        }
        self.redis_client.setex(
            key,
            timedelta(seconds=self.ttl),
            json.dumps(value)
        )
        logger.info(f"Cached result for key: {key}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        info = self.redis_client.info("stats")
        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / 
                       (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
        }
