# routers/image_count_router.py
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.schemas.common import ImageCountCreate, ImageCountInDB
from app.services.image_count_service import ImageCountService
from app.db.base import session_manager

router = APIRouter()

async def get_db_session():
    async with session_manager.create_session() as session:
        yield session

def get_image_count_service(db: AsyncSession = Depends(get_db_session)):
    return ImageCountService(db)

@router.post("/", response_model=ImageCountInDB, status_code=status.HTTP_201_CREATED)
async def create_or_update_image_count(data: ImageCountCreate, service: ImageCountService = Depends(get_image_count_service)):
    return await service.insert_or_update_image_count(data.image_path, data.face_count, data.processed_time)

@router.get("/", response_model=List[ImageCountInDB])
async def list_image_counts(service: ImageCountService = Depends(get_image_count_service)):
    return await service.get_all()