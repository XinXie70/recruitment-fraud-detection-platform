

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

_async_engine = create_async_engine(
    settings.async_database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout_seconds,
    pool_recycle=settings.db_pool_recycle_seconds,
)

AsyncSessionLocal = async_sessionmaker(
    _async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_db():
    """FastAPI dependency """
    async with AsyncSessionLocal() as session:
        yield session
