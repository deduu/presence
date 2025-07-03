# services/image_record_service.py
import os
import json

from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func
from datetime import datetime
import logging
from typing import Optional  # Import Optional

from app.services.base_service import BaseService
from app.db.models.attendance import ImageRecord, Face, Person, Image
from app.utils.file_store import PUBLIC_IMAGE_URL_PREFIX, SERVER_IMAGE_STORAGE_ROOT
logger = logging.getLogger(__name__)


class ImageRecordService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ImageRecord)

    def _get_image_url_from_record(self, record: ImageRecord) -> str:
        """Generate public URL from the associated Image model."""
        if record.image and record.image.image_path:
            filename = os.path.basename(record.image.image_path)
            return os.path.join(PUBLIC_IMAGE_URL_PREFIX, filename)
        return ""

    def _get_image_url_from_path(self, image_path: str) -> str:
        return os.path.join(PUBLIC_IMAGE_URL_PREFIX, os.path.basename(image_path))

    def _format_record(self, record: ImageRecord) -> dict:
        return {
            "record_id": record.record_id,
            "face_id": record.face_id,
            "detection_time": record.detection_time,
            "image_path": record.image.image_path if record.image else None,
            "image_url": self._get_image_url_from_record(record),
        }

    async def cleanup_orphan_images(self):
        stmt = (
            select(Image)
            .outerjoin(ImageRecord, Image.image_id == ImageRecord.image_id)
            .group_by(Image.image_id)
            .having(func.count(ImageRecord.record_id) == 0)
            .options(selectinload(Image.count))
        )
        result = await self.db.execute(stmt)
        orphan_images = result.scalars().all()

        logger.info(
            f"[cleanup_orphan_images] Found {len(orphan_images)} orphan images")

        for image in orphan_images:
            # Delete ImageCount first
            if image.count:
                logger.info(
                    f"[cleanup_orphan_images] Deleting ImageCount for image ID {image.image_id}")
                await self.db.delete(image.count)

            logger.info(
                f"[cleanup_orphan_images] Deleting orphan image ID {image.image_id}")
            await self.db.delete(image)

        if orphan_images:
            await self.db.commit()
            logger.info(
                "[cleanup_orphan_images] Orphan images deleted and committed")
        else:
            logger.info("[cleanup_orphan_images] No orphan images to delete")

    async def get_all(self, skip=0, limit=100):
        records = await super().get_all(skip, limit)
        return [self._format_record(r) for r in records]

    async def get_all_joined(self):
        stmt = (
            select(ImageRecord, Person.name)
            .join(Face, Face.face_id == ImageRecord.face_id)
            .join(Person, Person.person_id == Face.person_id, isouter=True)
            # <-- Ensure image is loaded
            .options(selectinload(ImageRecord.image))
        )
        result = await self.db.execute(stmt)
        return [
            {
                "record_id": record.record_id,
                "face_id": record.face_id,
                "image_path": record.image_path,
                "detection_time": record.detection_time,
                "image_url": self._get_image_url_from_record(record),
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
            # Ensure related Image is loaded
            .options(selectinload(ImageRecord.image))
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
                "image_id": record.image.image_id,
                "image_path": record.image.image_path,
                "image_url": self._get_image_url_from_path(record.image.image_path),
                "detection_time": record.detection_time,
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
        image_id: int,
        face_id: int,
        detection_time: datetime,
        face_location: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        batch_tag: Optional[str] = None
    ):
        try:
            record = ImageRecord(
                image_id=image_id,
                face_id=face_id,
                detection_time=detection_time,
                face_location=face_location,
                image_width=image_width,
                image_height=image_height,
                batch_tag=batch_tag
            )
            self.db.add(record)
            await self.db.commit()
            await self.db.refresh(record)
            return record
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error inserting image record for face ID {face_id}: {e}")
            raise

    async def insert_image_record_with_image(
        self,
        image_path: str,
        detection_time: datetime,
        face_id: int,
        face_location: str,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        batch_tag: Optional[str] = None,
    ) -> ImageRecord:
        # Check if the image already exists
        stmt = select(Image).where(Image.image_path == image_path)
        result = await self.db.execute(stmt)
        image = result.scalar_one_or_none()

        if not image:
            image = Image(image_path=image_path)
            self.db.add(image)
            # await self.db.commit()
            # await self.db.refresh(image)
            await self.db.flush()

        record = ImageRecord(
            image_id=image.image_id,
            face_id=face_id,
            detection_time=detection_time,
            face_location=face_location,
            image_width=image_width,
            image_height=image_height,
            batch_tag=batch_tag,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_by_person_id(self, person_id: int):
        try:
            stmt = (
                select(ImageRecord)
                .join(Face, Face.face_id == ImageRecord.face_id)
                .where(Face.person_id == person_id)
                .options(selectinload(ImageRecord.image))
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

    async def delete(self, record_id: int):
        try:
            # Step 1: Find the record to delete
            stmt = select(ImageRecord).where(
                ImageRecord.record_id == record_id)
            result = await self.db.execute(stmt)
            record = result.scalar_one_or_none()

            if not record:
                return False

            # Step 2: Delete the record
            await self.db.delete(record)
            await self.db.commit()

            # Step 3: Check and delete orphaned image if applicable
            await self.cleanup_orphan_images()

            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"[delete] Failed to delete ImageRecord {record_id}: {e}")
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

            await self.cleanup_orphan_images()
            return True
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error deleting image records for record ID {record_id}: {e}")
            raise
