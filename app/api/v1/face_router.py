# routers/face_router.py
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.schemas.common import FaceCreate, FaceInDB
from app.services.face_service import FaceService
from app.db.base import session_manager

router = APIRouter()

async def get_db_session():
    async with session_manager.create_session() as session:
        yield session

def get_face_service(db: AsyncSession = Depends(get_db_session)):
    return FaceService(db)

@router.post("/", response_model=FaceInDB, status_code=status.HTTP_201_CREATED)
async def create_face(face: FaceCreate, service: FaceService = Depends(get_face_service)):
    return await service.insert_new_face(face.face_encoding, face.first_seen)

@router.get("/", response_model=List[FaceInDB])
async def list_faces(service: FaceService = Depends(get_face_service)):
    ids, encodings = await service.get_all_known_faces()
    return [FaceInDB(face_id=id, face_encoding=enc, first_seen=None, last_seen=None) for id, enc in zip(ids, encodings)]
