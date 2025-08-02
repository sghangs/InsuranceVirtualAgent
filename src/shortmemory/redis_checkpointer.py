from typing import Any, Optional
from src.shortmemory.memory import RedisMemoryManager

class RedisCheckpointer:
    def __init__(self, redis_memory: RedisMemoryManager):
        self.memory = redis_memory

    async def get(self, key: str):
        return await self.memory.load_state(key)

    async def put(self, key: str, value):
        await self.memory.save_state(key, value)

    async def clear(self, key: str):
        await self.memory.delete_state(key)

    async def get_next_version(self, key: str) -> str:
        # Use Redis atomic increment for versioning
        version = await self.memory.client.incr(f"{key}:version")
        return str(version)