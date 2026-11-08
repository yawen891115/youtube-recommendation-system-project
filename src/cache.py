"""Redis-backed cache and fixed-window rate limit with a bounded local fallback."""

import asyncio
import json
import time
from collections import OrderedDict

from redis.asyncio import Redis
from redis.exceptions import RedisError


class CacheStore:
    def __init__(self, url=None, client=None, max_entries=2048):
        self.client = client or (
            Redis.from_url(
                url, decode_responses=True, socket_connect_timeout=0.3, socket_timeout=0.3
            )
            if url
            else None
        )
        self.max_entries = max_entries
        self.memory = OrderedDict()
        self.limits = {}
        self.lock = asyncio.Lock()
        self.connected = False

    async def ping(self):
        if self.client:
            try:
                self.connected = bool(await self.client.ping())
                return self.connected
            except (RedisError, OSError):
                self.connected = False
        return False

    @property
    def backend(self):
        return "redis" if self.connected else "memory"

    async def get(self, key):
        if self.client:
            try:
                value = await self.client.get(key)
                self.connected = True
                if value is not None:
                    return json.loads(value)
            except (RedisError, OSError, ValueError):
                self.connected = False
        async with self.lock:
            item = self.memory.get(key)
            if item and item[0] > time.monotonic():
                self.memory.move_to_end(key)
                return item[1]
            self.memory.pop(key, None)
        return None

    async def set(self, key, value, ttl=60):
        if self.client:
            try:
                await self.client.setex(key, ttl, json.dumps(value))
                self.connected = True
            except (RedisError, OSError):
                self.connected = False
        async with self.lock:
            self.memory[key] = (time.monotonic() + ttl, value)
            self.memory.move_to_end(key)
            while len(self.memory) > self.max_entries:
                self.memory.popitem(last=False)

    async def allow(self, client_id, limit):
        window = int(time.time()) // 60
        key = f"rate:{client_id}:{window}"
        if self.client:
            try:
                pipe = self.client.pipeline(transaction=True)
                pipe.incr(key)
                pipe.expire(key, 120)
                count, _ = await pipe.execute()
                self.connected = True
                return int(count) <= limit
            except (RedisError, OSError):
                self.connected = False
        async with self.lock:
            self.limits = {key: value for key, value in self.limits.items() if value[1] >= window}
            count = self.limits.get(client_id, (0, window))[0] + 1
            if client_id not in self.limits and len(self.limits) >= self.max_entries:
                return False
            self.limits[client_id] = (count, window)
            return count <= limit

    async def close(self):
        if self.client:
            await self.client.aclose()
