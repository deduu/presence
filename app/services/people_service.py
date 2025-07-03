import logging
import os
from typing import Any
from .base_service import BaseService
from sqlalchemy import update
from fastapi import UploadFile, HTTPException
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException

from urllib.parse import urlparse

from sqlalchemy import select, func, outerjoin
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.attendance import Person, Face
from app.schemas.person_schemas import PersonOut
from app.utils.file_store import save_person_image_async
from app.utils.file_store import SERVER_IMAGE_STORAGE_ROOT, SERVER_PERSON_IMAGE_ROOT
from app.services.face_service import FaceService

logger = logging.getLogger(__name__)


class PeopleService(BaseService):
    def __init__(self, db, face_service: FaceService):
        self.face_service = face_service

        super().__init__(db, Person)

    async def create(self, schema: Any) -> Any:
        name = schema.name
        dob = schema.date_of_birth
        contact = schema.contact_number

        try:
            # 1. Same contact number, different name → block
            stmt = select(Person).where(Person.contact_number == contact)
            result = await self.db.execute(stmt)
            person_by_contact = result.scalar_one_or_none()
            if person_by_contact and person_by_contact.name != name:
                raise HTTPException(
                    status_code=409,
                    detail="This contact number is already used by another person. Please use a different contact number.",
                )

            # 2. Same name AND contact number → block
            stmt = select(Person).where(
                Person.name == name,
                Person.contact_number == contact
            )
            result = await self.db.execute(stmt)
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=409,
                    detail="A person with the same name and contact number already exists.",
                )

            # 3. Same name AND date of birth → block
            if dob:
                stmt = select(Person).where(
                    Person.name == name,
                    Person.date_of_birth == dob
                )
                result = await self.db.execute(stmt)
                if result.scalar_one_or_none():
                    raise HTTPException(
                        status_code=409,
                        detail="A person with the same name and date of birth already exists.",
                    )

            # 4. Same contact number AND date of birth → block
            if dob:
                stmt = select(Person).where(
                    Person.contact_number == contact,
                    Person.date_of_birth == dob
                )
                result = await self.db.execute(stmt)
                if result.scalar_one_or_none():
                    raise HTTPException(
                        status_code=409,
                        detail="A person with the same contact number and date of birth already exists.",
                    )

            # ✅ Passed all checks — proceed with creation
            return await super().create(schema)

        except HTTPException:
            raise
        except IntegrityError:
            await self.db.rollback()
            logger.warning(
                f"Integrity error while creating {self.model.__name__}")
            raise HTTPException(
                status_code=409, detail="Duplicate person record")
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Unexpected error while creating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=500, detail="Internal Server Error")

    async def delete(self, person_id: int) -> bool:
        person = await self.get_by_id(person_id)

        # --- 1. Attempt to delete image file ---
        if person.image_path:
            try:
                # Convert URL to local path
                parsed = urlparse(person.image_path)
                # E.g., "/public_images/people/abc.jpg" → "app_data/permanent_images/people/abc.jpg"
                relative_path = parsed.path.replace(
                    "/public/people/images", "").lstrip("/")
                file_path = os.path.join(
                    SERVER_PERSON_IMAGE_ROOT, relative_path)

                abs_path = os.path.abspath(file_path)
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    logger.info(f"Deleted image file: {abs_path}")
                else:
                    logger.warning(
                        f"Image file not found for deletion: {abs_path}")
            except Exception as e:
                logger.warning(
                    f"Could not delete image for person {person_id}: {e}")

        # --- 2. Proceed with deleting from DB ---
        return await super().delete(person_id)

    async def upload_person_image(self, person_id: int, file: UploadFile):
        try:
            image_url, image_path = await save_person_image_async(file, person_id)

            stmt = (
                update(Person)
                .where(Person.person_id == person_id)
                .values(image_path=image_url)
            )

            await self.db.execute(stmt)
            await self.db.commit()

            # Add person image to face table
            await self.face_service.add_portrait_face(person_id, image_path, save_crop=True)

            return {"image_url": image_url}

        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=500, detail=f"Failed to upload image: {str(e)}"
            )

    async def list_with_face_counts(self, filters: dict = None):

        try:
            face_alias = aliased(Face)
            stmt = (
                select(Person, func.count(
                    face_alias.face_id).label("face_count"))
                .outerjoin(face_alias, Person.person_id == face_alias.person_id)
                .group_by(Person.person_id)
            )

            if filters:
                for key, value in filters.items():
                    stmt = stmt.where(getattr(Person, key).ilike(f"%{value}%"))

            result = await self.db.execute(stmt)
            rows = result.all()
            # print(f"image_path: {rows[0][0].__dict__}")
            # manually combine SQLAlchemy objects with extra field
            return [
                PersonOut.model_validate(row[0]).model_dump() | {
                    "face_count": row[1]}
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error listing people with face counts: {e}")
            raise
