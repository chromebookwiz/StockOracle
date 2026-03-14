from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


_RATE_LIMIT_LOCK = threading.Lock()
_LAST_CALL_BY_KEY: dict[str, float] = {}


def cache_directory() -> Path:
    root = os.getenv("STOCKORACLE_CACHE_DIR")
    if root:
        path = Path(root)
    else:
        path = Path(tempfile.gettempdir()) / "stockoracle-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{namespace}-{digest}"


def load_from_cache(key: str, ttl_seconds: int) -> Any | None:
    path = cache_directory() / f"{key}.pkl"
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > ttl_seconds:
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save_to_cache(key: str, value: Any) -> None:
    path = cache_directory() / f"{key}.pkl"
    with path.open("wb") as handle:
        pickle.dump(value, handle)


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