import hashlib
import json
import logging
from typing import Optional

from app.config import EMBEDDING_CACHE_TTL, LLM_CACHE_TTL, REDIS_URL

logger = logging.getLogger(__name__)

_redis_client: Optional[object] = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            from redis import Redis

            _redis_client = Redis.from_url(REDIS_URL, socket_connect_timeout=3, decode_responses=False)
            _redis_client.ping()
            logger.info("Redis connected: %s", REDIS_URL)
        except Exception:
            logger.warning("Redis unavailable at %s, caching disabled", REDIS_URL)
            _redis_client = False
    return _redis_client


def _hash_key(text: str) -> str:
    return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"


def cache_get_embedding(text: str) -> Optional[list[float]]:
    r = _get_redis()
    if not r:
        return None
    try:
        data = r.get(_hash_key(text))
        if data:
            return json.loads(data)
    except Exception:
        logger.debug("Redis get failed for embedding cache", exc_info=True)
    return None


def cache_set_embedding(text: str, embedding: list[float]) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(_hash_key(text), EMBEDDING_CACHE_TTL, json.dumps(embedding))
    except Exception:
        logger.debug("Redis set failed for embedding cache", exc_info=True)


def cache_get_llm_response(question: str, context_hash: str) -> Optional[str]:
    r = _get_redis()
    if not r:
        return None
    try:
        key = f"llm:{hashlib.sha256((question + context_hash).encode()).hexdigest()}"
        data = r.get(key)
        if data:
            return data.decode("utf-8")
    except Exception:
        logger.debug("Redis get failed for LLM cache", exc_info=True)
    return None


def cache_set_llm_response(question: str, context_hash: str, answer: str) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        key = f"llm:{hashlib.sha256((question + context_hash).encode()).hexdigest()}"
        r.setex(key, LLM_CACHE_TTL, answer.encode("utf-8"))
    except Exception:
        logger.debug("Redis set failed for LLM cache", exc_info=True)
