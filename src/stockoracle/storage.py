from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Protocol

import redis


class BinaryStore(Protocol):
    def get_bytes(self, key: str, ttl_seconds: int | None = None) -> bytes | None: ...

    def set_bytes(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...


class FileBinaryStore:
    def __init__(self, root_directory: str | None = None) -> None:
        configured = root_directory or os.getenv("STOCKORACLE_STORAGE_DIR")
        self.root = Path(configured) if configured else Path(tempfile.gettempdir()) / "stockoracle-store"
        self.root.mkdir(parents=True, exist_ok=True)

    def get_bytes(self, key: str, ttl_seconds: int | None = None) -> bytes | None:
        path = self.root / key
        if not path.exists():
            return None
        if ttl_seconds is not None:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > ttl_seconds:
                return None
        try:
            return path.read_bytes()
        except Exception:
            return None

    def set_bytes(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)


class RedisBinaryStore:
    def __init__(self, redis_url: str) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=False)

    def get_bytes(self, key: str, ttl_seconds: int | None = None) -> bytes | None:
        value = self.client.get(key)
        return bytes(value) if value is not None else None

    def set_bytes(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if ttl_seconds is None:
            self.client.set(key, value)
        else:
            self.client.setex(key, ttl_seconds, value)


_STORE: BinaryStore | None = None


def get_binary_store() -> BinaryStore:
    global _STORE
    if _STORE is not None:
        return _STORE

    redis_url = os.getenv("STOCKORACLE_REDIS_URL")
    if redis_url:
        _STORE = RedisBinaryStore(redis_url)
    else:
        _STORE = FileBinaryStore()
    return _STORE