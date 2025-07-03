# Updated: services/image_count_service.py
from app.services.base_service import BaseService
from app.db.models.attendance import ImageCount, Image
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageCountService (BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ImageCount)

    async def insert_or_update_image_count(self, image_path: str, face_count: int, processed_time: datetime):

        # Find image_id from image_path
        stmt = select(Image).where(Image.image_path == image_path)
        res = await self.db.execute(stmt)
        image = res.scalar_one_or_none()
        if not image:
            raise ValueError(f"Image not found for path: {image_path}")

        image_id = image.image_id
        logger.debug(
            f"IC -> path={image_path}  found image_id={image.image_id if image else None}")
        # Check if count exists
        stmt = select(ImageCount).where(ImageCount.image_id == image_id)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.face_count = face_count
            existing.processed_time = processed_time
        else:
            count = ImageCount(
                image_id=image_id,
                face_count=face_count,
                processed_time=processed_time,
            )
            self.db.add(count)
            await self.db.flush()
        await self.db.commit()

    async def insert_or_update_by_id(self, image_id: int, face_count: int, processed_time: datetime):
        stmt = select(ImageCount).where(ImageCount.image_id == image_id)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.face_count = face_count
            existing.processed_time = processed_time
        else:
            count = ImageCount(
                image_id=image_id,
                face_count=face_count,
                processed_time=processed_time,
            )
            self.db.add(count)

        await self.db.commit()

    async def insert_or_update_by_path(self, image_path: str, face_count: int, processed_time: datetime) -> ImageCount:
        try:
            # Get image by path
            image_stmt = select(Image).where(Image.image_path == image_path)
            image_result = await self.db.execute(image_stmt)
            image = image_result.scalar_one_or_none()

            if not image:
                image = Image(image_path=image_path)
                self.db.add(image)
                await self.db.flush()  # get image_id

            stmt = select(ImageCount).where(
                ImageCount.image_id == image.image_id)
            result = await self.db.execute(stmt)
            count = result.scalar_one_or_none()

            if count:
                count.face_count = face_count
                count.processed_time = processed_time
            else:
                count = ImageCount(
                    image_id=image.image_id,
                    face_count=face_count,
                    processed_time=processed_time
                )
                self.db.add(count)

            await self.db.commit()
            await self.db.refresh(count)
            return count

        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed to insert/update ImageCount for path={image_path}: {e}")
            raise

    async def get_by_image_id(self, image_id: int) -> Optional[ImageCount]:
        try:
            stmt = select(ImageCount).where(ImageCount.image_id == image_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                f"Failed to get ImageCount by image_id={image_id}: {e}")
            raise

    async def delete_by_image_id(self, image_id: int) -> bool:
        try:
            stmt = select(ImageCount).where(ImageCount.image_id == image_id)
            result = await self.db.execute(stmt)
            count = result.scalar_one_or_none()
            if count:
                await self.db.delete(count)
                await self.db.commit()
                return True
            return False
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Failed to delete ImageCount by image_id={image_id}: {e}")
            raise

    async def list_all(self) -> List[ImageCount]:
        try:
            stmt = select(ImageCount).options(selectinload(ImageCount.image))
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Failed to list ImageCounts: {e}")
            raise
