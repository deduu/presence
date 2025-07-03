# services/image_record_service.py
import os
import json

from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import logging
from typing import Optional  # Import Optional

from app.services.base_service import BaseService
from app.db.models.attendance import ImageRecord, Face, Person
from app.utils.file_store import PUBLIC_IMAGE_URL_PREFIX
logger = logging.getLogger(__name__)


class ImageRecordService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ImageRecord)

    def _get_image_url_from_path(self, image_path: str) -> str:
        """Convert server file path to public image URL."""
        filename = os.path.basename(image_path)

        return os.path.join(PUBLIC_IMAGE_URL_PREFIX, filename)

    def _format_record(self, record: ImageRecord) -> dict:
        """Convert DB model into dict with proper public-facing image_url."""
        return {
            "record_id": record.record_id,
            "face_id": record.face_id,
            "detection_time": record.detection_time,
            "image_path": record.image_path,

            "image_url": self._get_image_url_from_path(record.image_path),
        }

    async def get_all(self, skip=0, limit=100):
        records = await super().get_all(skip, limit)
        return [self._format_record(r) for r in records]

    async def get_all_joined(self):
        stmt = (
            select(ImageRecord, Person.name)
            .join(Face, Face.face_id == ImageRecord.face_id)
            .join(Person, Person.person_id == Face.person_id, isouter=True)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "record_id": record.record_id,
                "face_id": record.face_id,
                "image_path": record.image_path,
                "detection_time": record.detection_time,
                "image_url": self._get_image_url_from_path(record.image_path),
                "person_name": person_name or "Anonymous",
                "batch_tag": record.batch_tag,
            }
            for record, person_name in result.all()
        ]

    async def get_all_joined_filtered(self, person=None, start_time=None, end_time=None, batch_tag=None):
        stmt = (
            select(ImageRecord, Person.name)
            .join(Face, Face.face_id == ImageRecord.face_id)
            .join(Person, Person.person_id == Face.person_id, isouter=True)
        )

        if person:
            stmt = stmt.where(Person.name.ilike(f"%{person}%"))
        if start_time:
            stmt = stmt.where(ImageRecord.detection_time >= start_time)
        if end_time:
            stmt = stmt.where(ImageRecord.detection_time <= end_time)
        if batch_tag:
            stmt = stmt.where(ImageRecord.batch_tag.ilike(f"%{batch_tag}%"))

        result = await self.db.execute(stmt)
        return [
            {
                "record_id": record.record_id,
                "face_id": record.face_id,
                "image_path": record.image_path,
                "detection_time": record.detection_time,
                "image_url": self._get_image_url_from_path(record.image_path),
                "face_location": json.loads(record.face_location or "[0,0,0,0]"),
                "image_width": record.image_width,
                "image_height": record.image_height,
                "person_name": person_name or "Anonymous",
                "batch_tag": record.batch_tag,
            }
            for record, person_name in result.all()
        ]

    async def insert_image_record(
        self,
        image_path: str,
        face_id: int,
        detection_time: datetime,
        face_location: Optional[str] = None,  # Add this parameter
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        batch_tag: Optional[str] = None
    ):
        try:
            record = ImageRecord(
                image_path=image_path,
                face_id=face_id,
                detection_time=detection_time,
                face_location=face_location,  # Assign the new parameter
                image_width=image_width,
                image_height=image_height,
                batch_tag=batch_tag
            )
            self.db.add(record)
            await self.db.commit()
            await self.db.refresh(
                record
            )  # It's good practice to refresh after commit if you need the ID
            return record
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error inserting image record for face ID {face_id}: {e}")
            raise

    async def get_by_person_id(self, person_id: int):
        try:
            stmt = (
                select(ImageRecord)
                .join(Face, Face.face_id == ImageRecord.face_id)
                .where(Face.person_id == person_id)
            )
            result = await self.db.execute(stmt)
            records = result.scalars().all()
            # Format the records for the frontend
            reformat_records = [self._format_record(r) for r in records]

            return [self._format_record(r) for r in records]
        except Exception as e:
            logger.error(
                f"Error fetching image records for person ID {person_id}: {e}")
            raise

    async def delete_by_person_id(self, person_id: int):
        try:
            stmt = select(ImageRecord).where(
                ImageRecord.person_id == person_id)
            result = await self.db.execute(stmt)
            records = result.scalars().all()
            for record in records:
                await self.db.delete(record)
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error deleting image records for person ID {person_id}: {e}")
            raise

    async def delete_by_face_id(self, face_id: int):
        try:
            stmt = select(ImageRecord).where(ImageRecord.face_id == face_id)
            result = await self.db.execute(stmt)
            records = result.scalars().all()
            for record in records:
                await self.db.delete(record)
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error deleting image records for face ID {face_id}: {e}")
            raise

    async def delete_by_record_id(self, record_id: int):
        try:
            stmt = select(ImageRecord).where(
                ImageRecord.record_id == record_id)
            result = await self.db.execute(stmt)
            records = result.scalars().all()
            for record in records:
                await self.db.delete(record)
            await self.db.commit()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error deleting image records for record ID {record_id}: {e}")
            raise
