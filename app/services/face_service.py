# services/face_service.py

import numpy as np
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select
from typing import List, Tuple, Optional

from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload

from app.services.base_service import BaseService
from app.db.models.attendance import Face, Person, ImageRecord

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
    
    async def list_joined(self, known: str | None = None, person_id: int | None = None):
        stmt = (
            select(Face, Person.name)
            .join(Person, Face.person_id == Person.person_id, isouter=True)
        )
        if known == "known":
            stmt = stmt.where(Face.person_id.is_not(None))
        elif known == "anonymous":
            stmt = stmt.where(Face.person_id.is_(None))
        if person_id:
            stmt = stmt.where(Face.person_id == person_id)

        res = await self.db.execute(stmt)
        return [
            dict(
                face_id=f.face_id,
                first_seen=f.first_seen,
                last_seen=f.last_seen,
                person_id=f.person_id,
                person_name=name,
            )
            for f, name in res.all()
        ]

    async def get_detail(self, face_id: int):
        stmt = select(Face).options(joinedload(Face.person)).where(Face.face_id == face_id)
        res  = await self.db.execute(stmt)
        face = res.scalar_one_or_none()
        if not face:
            return None
        return {
            "face_id": face.face_id,
            "first_seen": face.first_seen,
            "last_seen":  face.last_seen,
            "person_id":  face.person_id,
            "person_name": face.person.name if face.person else None,
        }

    async def records_for_face(self, face_id: int):
        stmt = (
            select(ImageRecord)
            .where(ImageRecord.face_id == face_id)
            .order_by(ImageRecord.detection_time.desc())
        )
        res = await self.db.execute(stmt)
        return res.scalars().all()

    async def associate(self, face_id: int, person_id: int):
        face = await self.get_by_id(face_id)
        face.person_id = person_id
        await self.db.commit()

    async def disassociate(self, face_id: int):
        face = await self.get_by_id(face_id)
        face.person_id = None
        await self.db.commit()