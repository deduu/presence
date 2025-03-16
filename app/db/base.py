import asyncio, contextlib, logging, os, sys, time
from dotenv import load_dotenv, dotenv_values
from typing import AsyncIterator, Any, Dict, List, Optional, Tuple, Type, Union
# from app.db.models.models import Base

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import logging

logger = logging.getLogger(__name__)

Base = declarative_base()
# PostgreSQL connection
DATABASE_URL = "postgresql+asyncpg://postgres:admin@localhost:5432/Presence"
engine = create_async_engine(DATABASE_URL, echo=True)  # Timeout before a new connection attempt

async_session = async_sessionmaker(bind=engine, expire_on_commit=False)
async def create_async_db():
    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class DatabaseSessionManager:
    def __init__(self, sessionmaker, engine):
        self.sessionmaker = sessionmaker
        self.engine = engine

    async def close(self):
        if(self.engine):
            await self.engine.dispose()
            
    @contextlib.asynccontextmanager
    async def create_session(self) ->AsyncIterator[AsyncSession]:
        session = self.sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

session_manager = DatabaseSessionManager(async_session, engine)

async def get_async_db():
    async with session_manager.create_session() as session:
        try:
            yield session
        finally:
            await session.close()





