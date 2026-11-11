"""
Async PostgreSQL engine and session factory.

All database I/O in the application uses these helpers.
Sync variants are provided for seeding scripts and Alembic migrations.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Async engine (used by FastAPI routes and agents)
# --------------------------------------------------------------------------- #

_async_engine = None
_AsyncSessionFactory = None


def _get_async_engine():
    global _async_engine
    if _async_engine is None:
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Add DATABASE_URL=postgresql+asyncpg://user:pass@host/db to .env"
            )
        url = settings.database_url
        # Ensure asyncpg driver is in the URL
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        _async_engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    return _async_engine


def _get_async_session_factory():
    global _AsyncSessionFactory
    if _AsyncSessionFactory is None:
        _AsyncSessionFactory = async_sessionmaker(
            _get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionFactory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a scoped database session."""
    factory = _get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# --------------------------------------------------------------------------- #
# Sync engine (used by seed scripts and Alembic migrations)
# --------------------------------------------------------------------------- #

_sync_engine = None
_SyncSessionFactory = None


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        url = settings.database_url or ""
        # Convert asyncpg URL back to standard psycopg2
        url = url.replace("postgresql+asyncpg://", "postgresql://")
        url = url.replace("postgresql+psycopg2://", "postgresql://")
        _sync_engine = create_engine(url, pool_pre_ping=True, echo=False)
    return _sync_engine


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Sync context manager for seeding and migration scripts."""
    factory = sessionmaker(_get_sync_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def init_db() -> None:
    """Create all tables (used in tests and first-run setups without Alembic)."""
    from src.db.models import Base
    engine = _get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def drop_db() -> None:
    """Drop all tables (tests only)."""
    from src.db.models import Base
    engine = _get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
