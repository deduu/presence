# services/image_count_service.py

from app.services.base_service import BaseService
from app.db.models.attendance import ImageCount
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImageCountService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ImageCount)
    
    async def insert_or_update_image_count(self, image_path: str, face_count: int, processed_time: datetime):
        try:
            stmt = select(ImageCount).where(ImageCount.image_path == image_path)
            result = await self.db.execute(stmt)
            record = result.scalar_one_or_none()
            if record:
                record.face_count = face_count
                record.processed_time = processed_time
            else:
                record = ImageCount(
                    image_path=image_path,
                    face_count=face_count,
                    processed_time=processed_time
                )
                self.db.add(record)
            await self.db.commit()
            return record
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error inserting/updating image count for {image_path}: {e}")
            raise


