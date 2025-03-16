# database.py

import logging
from typing import List, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from sqlalchemy import insert, update
import asyncio
from app.db.base import *
from app.db.models.attendance import Face, ImageRecord, ImageCount


logger = logging.getLogger(__name__)

class AsyncDatabaseHandler:
    def __init__(self):
        self.session_manager = session_manager  # Use the session manager you provided

    async def connect(self):
        # Connection is managed by session_manager
        pass

    async def close(self):
        await self.session_manager.close()

    async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
        """
        Returns two lists: known_face_ids and known_face_encodings
        """
        try:
            async with self.session_manager.create_session() as session:
                result = await session.execute(select(Face))
                faces = result.scalars().all()
                known_face_ids = [face.face_id for face in faces]
                known_face_encodings = [np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces]
                return known_face_ids, known_face_encodings
        except Exception as e:
            logger.error(f"Error retrieving known faces: {e}")
            return [], []

    async def insert_new_face(self, face_encoding: np.ndarray, current_time) -> Optional[int]:
        """
        Inserts a new face into the database and returns the face_id
        """
        try:
            face_encoding_bytes = face_encoding.tobytes()
            new_face = Face(
                face_encoding=face_encoding_bytes,
                first_seen=current_time,
                last_seen=current_time
            )
            async with self.session_manager.create_session() as session:
                session.add(new_face)
                await session.commit()
                await session.refresh(new_face)
                logger.info(f"Inserted new face with ID {new_face.face_id}")
                return new_face.face_id
        except Exception as e:
            logger.error(f"Error inserting new face: {e}")
            return None

    async def update_last_seen(self, face_id: int, current_time, face_encoding: Optional[np.ndarray] = None):
        """
        Updates the last_seen time (and optionally the face_encoding) of a face
        """
        try:
            async with self.session_manager.create_session() as session:
                stmt = select(Face).where(Face.face_id == face_id)
                result = await session.execute(stmt)
                face = result.scalar_one_or_none()
                if face:
                    face.last_seen = current_time
                    if face_encoding is not None:
                        face.face_encoding = face_encoding.tobytes()
                    await session.commit()
                    logger.info(f"Updated last_seen for face ID {face_id}")
                else:
                    logger.error(f"No face found with ID {face_id}")
        except Exception as e:
            logger.error(f"Error updating last_seen: {e}")

    async def insert_image_record(self, image_path: str, face_id: int, detection_time):
        """
        Inserts a new image record into the database
        """
        try:
            image_record = ImageRecord(
                image_path=image_path,
                face_id=face_id,
                detection_time=detection_time
            )
            async with self.session_manager.create_session() as session:
                session.add(image_record)
                await session.commit()
                logger.info(f"Inserted image record for face ID {face_id} and image {image_path}")
        except Exception as e:
            logger.error(f"Error inserting image record: {e}")

    async def insert_or_update_image_count(self, image_path: str, face_count: int, processed_time):
        """
        Inserts or updates the image count for an image
        """
        try:
            async with self.session_manager.create_session() as session:
                stmt = select(ImageCount).where(ImageCount.image_path == image_path)
                result = await session.execute(stmt)
                image_count = result.scalar_one_or_none()
                if image_count:
                    image_count.face_count = face_count
                    image_count.processed_time = processed_time
                    logger.info(f"Updated image count for {image_path}")
                else:
                    image_count = ImageCount(
                        image_path=image_path,
                        face_count=face_count,
                        processed_time=processed_time
                    )
                    session.add(image_count)
                    logger.info(f"Inserted image count for {image_path}")
                await session.commit()
        except Exception as e:
            logger.error(f"Error inserting/updating image count: {e}")

    # Additional CRUD methods can be added here as needed
