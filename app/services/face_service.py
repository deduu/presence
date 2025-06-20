# services/face_service.py

from app.services.base_service import BaseService
from app.db.models.attendance import Face
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from sqlalchemy import select
from typing import List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FaceService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db, Face)

    
    async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
        try:
            result = await self.db.execute(select(Face))
            faces = result.scalars().all()
            known_face_ids = [face.face_id for face in faces]
            known_face_encodings = [np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces]
            return known_face_ids, known_face_encodings
        except Exception as e:
            logger.error(f"Error retrieving known faces: {e}")
            return [], []
    
    async def insert_new_face(self, encoding: np.ndarray, current_time: datetime):
        try:
            encoding_bytes = encoding.tobytes()
            face = Face(face_encoding=encoding_bytes, first_seen=current_time, last_seen=current_time)
            self.db.add(face)
            await self.db.commit()
            await self.db.refresh(face)
            return face
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error inserting new face: {e}")
            raise

    async def update_last_seen(self, face_id: int, current_time: datetime, encoding: Optional[np.ndarray] = None):
        try:
            face = await self.get_by_id(face_id)
            face.last_seen = current_time
            if encoding is not None:
                face.face_encoding = encoding.tobytes()
            await self.db.commit()
            await self.db.refresh(face)
            return face
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating last_seen for face ID {face_id}: {e}")
            raise
