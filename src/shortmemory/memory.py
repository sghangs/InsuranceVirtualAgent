import os
import redis.asyncio as redis
import json
from redis.connection import SSLConnection
from dotenv import load_dotenv

load_dotenv()

class RedisMemoryManager:
    def __init__(self):
        self.pool = redis.ConnectionPool(
            host=os.getenv('REDIS_HOST', 'your-production-redis-host'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            password=os.getenv('REDIS_PASS', 'your-password'),
            ssl=os.getenv('REDIS_SSL', 'true').lower() == 'true',
            ssl_cert_reqs=None,
            decode_responses=True,
            connection_class=SSLConnection if os.getenv('REDIS_SSL', 'true').lower() == 'true' else redis.connection.Connection
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def _key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:short_term_memory"

    async def save_state(self, thread_id: str, state: dict, ttl: int = 3600):
        await self.client.set(self._key(thread_id), json.dumps(state), ex=ttl)

    async def load_state(self, thread_id: str) -> dict:
        raw = await self.client.get(self._key(thread_id))
        return json.loads(raw) if raw else {}

    async def delete_state(self, thread_id: str):
        await self.client.delete(self._key(thread_id))
