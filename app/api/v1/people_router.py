from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.schemas.person_schemas import PersonOut, PersonCreate, PersonUpdate
from app.services.people_service import PeopleService
from app.services.face_service import FaceService
from app.services.image_record_service import ImageRecordService
from app.services.image_count_service import ImageCountService
from app.db.base import session_manager

router = APIRouter()


async def get_db_session():
    async with session_manager.create_session() as session:
        yield session


def get_image_record_service(session: AsyncSession = Depends(get_db_session)):
    return ImageRecordService(session)


def get_image_count_service(session: AsyncSession = Depends(get_db_session)):
    return ImageCountService(session)


async def get_face_service(session: AsyncSession = Depends(get_db_session), image_record_service: ImageRecordService = Depends(get_image_record_service), get_image_count_service: ImageCountService = Depends(get_image_count_service)):
    return FaceService(session, image_record_service, get_image_count_service)


async def svc(session: AsyncSession = Depends(get_db_session), face_service: FaceService = Depends(get_face_service)):
    return PeopleService(session, face_service)


@router.get("/", response_model=List[PersonOut])
async def list_people(q: str | None = None, service: PeopleService = Depends(svc)):
    filters = {"name": q} if q else None
    return await service.list_with_face_counts(filters)


@router.post("/", response_model=PersonOut, status_code=201)
async def create_person(body: PersonCreate, service: PeopleService = Depends(svc)):
    return await service.create(body)


@router.get("/{person_id}", response_model=PersonOut)
async def get_person(person_id: int, service: PeopleService = Depends(svc)):
    return await service.get(person_id)


@router.put("/{person_id}", response_model=PersonOut)
async def update_person(person_id: int, body: PersonUpdate, service: PeopleService = Depends(svc)):
    return await service.update(person_id, body)


@router.delete("/{person_id}", status_code=204)
async def delete_person(person_id: int, service: PeopleService = Depends(svc)):
    await service.delete(person_id)


@router.post("/{person_id}/image", status_code=status.HTTP_200_OK)
async def upload_person_image(
    person_id: int,
    file: UploadFile = File(...),
    service: PeopleService = Depends(svc)

):
    return await service.upload_person_image(person_id, file)
