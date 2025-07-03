# routers/face_router.py
from app.schemas.common import FaceCreate, FaceInDB
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from app.db.base import session_manager
from app.services.face_service import FaceService
from app.services.image_record_service import ImageRecordService
from app.schemas.common import FaceOut, ImageRecordOut


async def get_db_session():
    async with session_manager.create_session() as session:
        yield session


def get_image_record_service(db: AsyncSession = Depends(get_db_session)):
    return ImageRecordService(db)


router = APIRouter()


# ---- dependency ----
async def svc(
    session: AsyncSession = Depends(get_db_session),
    image_record_service: ImageRecordService = Depends(
        get_image_record_service),
):
    return FaceService(session, image_record_service)


# ---- bodies ----
class _AssociateBody(BaseModel):
    person_id: int


# ---- endpoints ----
@router.get("/", response_model=List[FaceOut])
async def list_faces(
    filter: Optional[str] = None,
    person_id: Optional[int] = None,
    service: FaceService = Depends(svc),
):
    return await service.list_joined(filter, person_id)


@router.get("/{face_id}", response_model=FaceOut)
async def get_face(face_id: int, service: FaceService = Depends(svc)):
    face = await service.get_detail(face_id)
    if not face:
        raise HTTPException(404, "Face not found")
    return face


@router.get("/{face_id}/records", response_model=List[ImageRecordOut])
async def face_records(face_id: int, service: FaceService = Depends(svc)):
    return await service.records_for_face(face_id)


@router.post("/{face_id}/associate", status_code=status.HTTP_204_NO_CONTENT)
async def associate_face(
    face_id: int, body: _AssociateBody, service: FaceService = Depends(svc)
):
    await service.associate(face_id, body.person_id)


@router.post("/{face_id}/associate-unique", status_code=status.HTTP_204_NO_CONTENT)
async def associate_face(
    face_id: int, body: _AssociateBody, service: FaceService = Depends(svc)
):
    await service.associate_transfer_then_delete(face_id, body.person_id)


@router.post("/{face_id}/disassociate", status_code=status.HTTP_204_NO_CONTENT)
async def disassociate_face(face_id: int, service: FaceService = Depends(svc)):
    await service.disassociate(face_id)


@router.delete("/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face(face_id: int, service: FaceService = Depends(svc)):
    await service.delete(face_id)
