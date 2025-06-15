import asyncio, contextlib, logging, os, sys, time
from dotenv import load_dotenv, dotenv_values
from typing import AsyncIterator, AsyncGenerator,Any, Dict, List, Optional, Tuple, Type, Union
# from app.db.models.models import Base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings


import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


# PostgreSQL connection
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,         # Minimum number of persistent connections in the pool
    max_overflow=15      # Allow up to 15 additional connections (total up to 20)
)

# Create an async session factory bound to the engine.
# Create a factory for async sessions
async_session = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False
)

async def create_async_db():
    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Custom session manager that uses an async context manager.
class DatabaseSessionManager:
    def __init__(self, sessionmaker, engine):
        self.sessionmaker = sessionmaker
        self.engine = engine

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            
    @contextlib.asynccontextmanager
    async def create_session(self) -> AsyncIterator[AsyncSession]:
        # Every call creates a new session that gets its own connection from the pool.
        session = self.sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Instantiate the session manager.
session_manager = DatabaseSessionManager(async_session, engine)

# Optional helper if you wish to iterate over sessions.
async def get_async_db():
    async with session_manager.create_session() as session:
        try:
            yield session
        finally:
            await session.close()


# class DatabaseSessionManager:
#     """Manages asynchronous SQLAlchemy database sessions."""
#     def __init__(self, database_url: str):
#         self.engine = create_async_engine(database_url, echo=settings.DEBUG_MODE, pool_size=5,         # Minimum number of persistent connections in the pool
#     max_overflow=15 )
#         # Configure async_sessionmaker to use AsyncSession
#         self.async_session_factory = async_sessionmaker(
#             self.engine,
#             expire_on_commit=False, # Do not expire objects after commit
#             class_=AsyncSession,    # Use AsyncSession for async operations
#             autoflush=False         # Do not autoflush changes to the database
#         )

#     async def create_all_tables(self):
#         """Creates all tables defined in Base metadata."""
#         async with self.engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
#             print("Database tables created or already exist.")

#     async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
#         """Dependency to get an asynchronous database session."""
#         async with self.async_session_factory() as session:
#             try:
#                 yield session
#             except Exception:
#                 await session.rollback()
#                 raise
#             finally:
#                 await session.close()


# db_manager = DatabaseSessionManager(settings.DATABASE_URL)
