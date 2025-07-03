# routers/image_record_router.py
import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from app.schemas.image_record import ImageRecordCreate, ImageRecordInDB, ImageRecordOut
from app.services.image_record_service import ImageRecordService
from app.db.base import session_manager

router = APIRouter()

logger = logging.getLogger(__name__)


async def get_db_session():
    async with session_manager.create_session() as session:
        yield session


def get_image_record_service(db: AsyncSession = Depends(get_db_session)):
    return ImageRecordService(db)


@router.post("/", response_model=ImageRecordInDB, status_code=status.HTTP_201_CREATED)
async def create_image_record(
    record: ImageRecordCreate,
    service: ImageRecordService = Depends(get_image_record_service),
):
    return await service.insert_image_record(
        record.image_path, record.face_id, record.detection_time
    )


@router.get("/", response_model=List[ImageRecordOut])
async def list_image_records(
    person: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    batch_tag: Optional[str] = None,
    service: ImageRecordService = Depends(get_image_record_service),
):
    return await service.get_all_joined_filtered(person, start_time, end_time, batch_tag)


# @router.get("/", response_model=List[ImageRecordOut])
# async def list_image_records(
#     service: ImageRecordService = Depends(get_image_record_service),
# ):
#     return await service.get_all_joined()


@router.get("/by-person/{person_id}", response_model=List[ImageRecordInDB])
async def list_by_person(
    person_id: int, service: ImageRecordService = Depends(get_image_record_service)
):

    return await service.get_by_person_id(person_id)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_records(
    record_ids: list[int],
    service: ImageRecordService = Depends(get_image_record_service),
):
    for rid in record_ids:
        await service.delete_by_record_id(rid)
    return {"status": "deleted", "count": len(record_ids)}
