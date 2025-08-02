from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from models import Base
from dotenv import load_dotenv
load_dotenv()

# Database configuration
DATABASE_URL = "postgresql+asyncpg://{0}:{1}@{2}:{3}/{4}".format(
    os.getenv("PG_USER"),
    os.getenv("PG_PASSWORD"),
    os.getenv("PG_HOST"),
    os.getenv("PG_PORT"),
    os.getenv("PG_DB")
)

# Create the SQLAlchemy async engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Create all tables in the database asynchronously
async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Dependency to get the async database session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session