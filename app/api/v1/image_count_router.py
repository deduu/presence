from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.schemas.image_count import ImageCountInDB
from app.schemas.image_count import ImageCountCreate, ImageCountUpdate
from app.services.image_count_service import ImageCountService
from app.db.base import session_manager

router = APIRouter()


async def get_db_session():
    async with session_manager.create_session() as session:
        yield session


def get_image_count_service(db: AsyncSession = Depends(get_db_session)):
    return ImageCountService(db)


@router.post("/", response_model=ImageCountInDB, status_code=status.HTTP_201_CREATED)
async def create_or_update_by_path(data: ImageCountCreate, service: ImageCountService = Depends(get_image_count_service)):
    return await service.insert_or_update_by_path(
        image_path=data.image_path,
        face_count=data.face_count,
        processed_time=data.processed_time
    )


@router.get("/", response_model=List[ImageCountInDB])
async def list_image_counts(service: ImageCountService = Depends(get_image_count_service)):
    return await service.list_all()


@router.get("/{image_id}", response_model=ImageCountInDB)
async def get_image_count(image_id: int, service: ImageCountService = Depends(get_image_count_service)):
    return await service.get(image_id)


@router.put("/{image_id}", response_model=ImageCountInDB)
async def update_image_count(
    image_id: int,
    data: ImageCountUpdate,
    service: ImageCountService = Depends(get_image_count_service)
):
    return await service.update(image_id, data)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_count(image_id: int, service: ImageCountService = Depends(get_image_count_service)):
    await service.delete(image_id)
