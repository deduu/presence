# services/image_count_service.py

from app.db.models.attendance import ImageCount
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageCountService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_or_update_by_record_id(self, record_id: int, face_count: int, processed_time: datetime) -> ImageCount:
        """Insert or update ImageCount by referencing the ImageRecord ID."""
        try:
            stmt = select(ImageCount).where(ImageCount.record_id == record_id)
            result = await self.db.execute(stmt)
            record = result.scalar_one_or_none()

            if record:
                record.face_count = face_count
                record.processed_time = processed_time
            else:
                record = ImageCount(
                    record_id=record_id,
                    face_count=face_count,
                    processed_time=processed_time
                )
                self.db.add(record)

            await self.db.commit()
            await self.db.refresh(record)
            return record
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed insert/update ImageCount for record_id={record_id}: {e}")
            raise

    async def get_by_record_id(self, record_id: int) -> Optional[ImageCount]:
        try:
            stmt = select(ImageCount).where(ImageCount.record_id == record_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                f"Failed to retrieve ImageCount by record_id={record_id}: {e}")
            raise

    async def delete_by_record_id(self, record_id: int) -> bool:
        try:
            stmt = select(ImageCount).where(ImageCount.record_id == record_id)
            result = await self.db.execute(stmt)
            record = result.scalar_one_or_none()
            if record:
                await self.db.delete(record)
                await self.db.commit()
                return True
            return False
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed to delete ImageCount by record_id={record_id}: {e}")
            raise

    async def list_all(self) -> List[ImageCount]:
        try:
            stmt = select(ImageCount)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to list all ImageCounts: {e}")
            raise
