from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time
from typing import Any, Callable, TypeVar

from .storage import get_binary_store


T = TypeVar("T")


_RATE_LIMIT_LOCK = threading.Lock()
_LAST_CALL_BY_KEY: dict[str, float] = {}


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{namespace}-{digest}"


def load_from_cache(key: str, ttl_seconds: int) -> Any | None:
    payload = get_binary_store().get_bytes(f"cache/{key}.pkl", ttl_seconds=ttl_seconds)
    if payload is None:
        return None
    try:
        return pickle.loads(payload)
    except Exception:
        return None


def save_to_cache(key: str, value: Any) -> None:
    get_binary_store().set_bytes(f"cache/{key}.pkl", pickle.dumps(value))


def rate_limit(key: str, minimum_interval_seconds: float) -> None:
    with _RATE_LIMIT_LOCK:
        previous = _LAST_CALL_BY_KEY.get(key, 0.0)
        now = time.monotonic()
        wait_seconds = minimum_interval_seconds - (now - previous)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_CALL_BY_KEY[key] = time.monotonic()


def cached_call(
    namespace: str,
    payload: dict[str, Any],
    ttl_seconds: int,
    limiter_key: str,
    minimum_interval_seconds: float,
    loader: Callable[[], T],
) -> T:
    key = cache_key(namespace, payload)
    cached = load_from_cache(key, ttl_seconds)
    if cached is not None:
        return cached
    rate_limit(limiter_key, minimum_interval_seconds)
    value = loader()
    save_to_cache(key, value)
    return value