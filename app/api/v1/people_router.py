from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.schemas.person_schemas import PersonOut, PersonCreate, PersonUpdate
from app.services.people_service import PeopleService
from app.db.base import session_manager

router = APIRouter()

async def get_db_session():
    async with session_manager.create_session() as session:
        yield session

async def svc(session: AsyncSession = Depends(get_db_session)):
    return PeopleService(session)

@router.get("/", response_model=List[PersonOut])
async def list_people(q: str | None = None, service: PeopleService = Depends(svc)):
    if q:
        return await service.list({"name": q})
    return await service.list()

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
