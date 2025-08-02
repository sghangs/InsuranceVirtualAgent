import sys
import asyncio
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async
import threading
from src.exception.exception import InsuranceAgentException

class ConnectionManager:
    _instance_lock = threading.Lock()

    def __init__(self, db_config):
        self.db_config = db_config
        print("Printing DB Config")
        print(self.db_config)
        self.pool = AsyncConnectionPool(
            conninfo=(
                f"host={db_config['host']} "
                f"port={db_config['port']} "
                f"dbname={db_config['database']} "
                f"user={db_config['user']} "
                f"password={db_config['password']}"
            ),
            min_size=db_config.get("minconn", 1),
            max_size=db_config.get("maxconn", 2),
            open=False  # Delay opening until explicitly called
        )

    async def open_pool(self):
        try:
            await self.pool.open(wait=True)
        except Exception as e:
            raise InsuranceAgentException("Failed to open async connection pool", error_obj=e)

    async def close_pool(self):
        try:
            await self.pool.close()
        except Exception as e:
            raise InsuranceAgentException("Failed to close async connection pool", error_obj=e)

    async def get_conn(self):
        try:
            conn = await self.pool.getconn()
            await register_vector_async(conn)
            return conn
        except Exception as e:
            raise InsuranceAgentException("Unable to get connection from pool", error_obj=e)

    async def put_conn(self, conn):
        try:
            if conn:
                await self.pool.putconn(conn)
        except Exception as e:
            raise InsuranceAgentException("Unable to return connection to pool", error_obj=e)

    @classmethod
    def instance(cls, db_config):
        if not hasattr(cls, "_instance"):
            with cls._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = cls(db_config)
        return cls._instance