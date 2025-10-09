import json
import logging
from typing import Optional, List
import redis.asyncio as aioredis
from src.app.config import settings
from src.app.schema import DEFAULT_WEATHER, WeatherDataPoint, WeatherPayload

log = logging.getLogger(__name__)

# Global Redis client instance
redis_client: Optional[aioredis.Redis] = None


async def init_redis():
    """Initialize global Redis client and test connection."""
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            encoding="utf-8",
            decode_responses=True,
        )

    try:
        # Test connection
        await redis_client.ping()
        log.info("Redis client successfully initialized and connected.")
    except Exception as e:
        log.exception("Redis ping failed: %s", e)
        # Re-raise the exception to stop the application startup if critical
        raise


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        log.info("Redis client connection closed.")


def _key_for(lat: float, lon: float, ymd: str, hour: str) -> str:
    """Generates the main key for a single hourly weather payload."""
    # Example: openmeteo:34.05,-118.24:20240720:10
    return f"openmeteo:{lat},{lon}:{ymd}:{hour}"


def _index_key(lat: float, lon: float, ymd: str) -> str:
    """Generates the key for the daily set index."""
    # Example: openmeteo:index:34.05,-118.24:20240720
    return f"openmeteo:index:{lat},{lon}:{ymd}"


async def get_cached(lat: float, lon: float, ymd: str, hour: str) -> Optional[WeatherPayload]:
    """Retrieves and deserializes a WeatherPayload from cache."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() first.")

    key = _key_for(lat, lon, ymd, hour)
    raw = await redis_client.get(key)
    
    if not raw:
        return None
        
    try:
        data = json.loads(raw)
        
        # Manually deserialize 'weather' list, applying defaults for backward compatibility
        if "weather" in data and isinstance(data["weather"], list):
            data["weather"] = [
                # Merge defaults first, then overlay with cached data 'w'
                WeatherDataPoint(**{**DEFAULT_WEATHER, **w}) for w in data["weather"]
            ]
            
        # Pydantic validates the whole structure
        return WeatherPayload(**data)
        
    except Exception as e:
        log.warning("Failed to decode cached data for %s: %s", key, e)
        # If decode fails, remove potentially corrupt key
        await redis_client.delete(key)
        return None


async def set_cached(lat: float, lon: float, ymd: str, hour: str, payload: WeatherPayload):
    """Cache a WeatherPayload and update daily index."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() first.")

    key = _key_for(lat, lon, ymd, hour)
    
    # Cache the data itself
    await redis_client.set(key, payload.model_dump_json())
    await redis_client.expire(key, settings.REDIS_TTL_SECONDS)

    # Update the daily index (a Redis Set)
    idx = _index_key(lat, lon, ymd)
    await redis_client.sadd(idx, key)
    # Set expiration on the index key as well
    await redis_client.expire(idx, settings.REDIS_INDEX_TTL_SECONDS)


async def get_daily_index(lat: float, lon: float, ymd: str) -> List[str]:
    """Return all hourly keys for a given day and location."""
    if redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() first.")

    idx = _index_key(lat, lon, ymd)
    # Redis Set members are returned as a set, convert to list for typing consistency
    return list(await redis_client.smembers(idx))
