# services/image_record_service.py


from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from typing import Optional  # Import Optional

from app.services.base_service import BaseService
from app.db.models.attendance import ImageRecord, Face

logger = logging.getLogger(__name__)


class ImageRecordService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ImageRecord)

    async def insert_image_record(
        self,
        image_path: str,
        face_id: int,
        detection_time: datetime,
        face_location: Optional[str] = None,  # Add this parameter
    ):
        try:
            record = ImageRecord(
                image_path=image_path,
                face_id=face_id,
                detection_time=detection_time,
                face_location=face_location,  # Assign the new parameter
            )
            self.db.add(record)
            await self.db.commit()
            await self.db.refresh(
                record
            )  # It's good practice to refresh after commit if you need the ID
            return record
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error inserting image record for face ID {face_id}: {e}")
            raise

    async def get_by_person_id(self, person_id: int):
        try:
            stmt = (
                select(ImageRecord)
                .join(Face, Face.face_id == ImageRecord.face_id)
                .where(Face.person_id == person_id)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching image records for person ID {person_id}: {e}")
            raise
