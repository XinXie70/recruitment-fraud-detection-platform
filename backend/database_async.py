"""
Async database session — built on SQLAlchemy 2.0 async engine + asyncpg.

Provides ``get_async_db`` for use with FastAPI ``async def`` endpoints that
want non-blocking database access.  Requires ``asyncpg`` (pip install asyncpg).

Usage::

    from database_async import get_async_db

    @router.get("/items")
    async def list_items(db: AsyncSession = Depends(get_async_db)):
        result = await db.execute(select(Model))
        return result.scalars().all()
"""

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
    """FastAPI dependency — yields an ``AsyncSession`` and closes it after."""
    async with AsyncSessionLocal() as session:
        yield session
