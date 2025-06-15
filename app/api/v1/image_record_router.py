# routers/image_record_router.py
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.schemas.common import ImageRecordCreate, ImageRecordInDB
from app.services.image_record_service import ImageRecordService
from app.db.base import session_manager

router = APIRouter(prefix="/image-records", tags=["Image Records"])

async def get_db_session():
    async with session_manager.create_session() as session:
        yield session

def get_image_record_service(db: AsyncSession = Depends(get_db_session)):
    return ImageRecordService(db)

@router.post("/", response_model=ImageRecordInDB, status_code=status.HTTP_201_CREATED)
async def create_image_record(record: ImageRecordCreate, service: ImageRecordService = Depends(get_image_record_service)):
    return await service.insert_image_record(record.image_path, record.face_id, record.detection_time)

@router.get("/", response_model=List[ImageRecordInDB])
async def list_image_records(service: ImageRecordService = Depends(get_image_record_service)):
    return await service.get_all()