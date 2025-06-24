from pydantic import BaseModel, ConfigDict
from datetime import date, datetime
from typing import Optional


class PersonBase(BaseModel):
    name: str
    date_of_birth: Optional[date] = None
    address: Optional[str] = None
    contact_number: Optional[str] = None


class PersonCreate(PersonBase):
    pass


class PersonUpdate(PersonBase):
    pass


class PersonOut(BaseModel):
    person_id: int
    name: str
    date_of_birth: Optional[date] = None  # ✅ changed from str to datetime
    address: Optional[str] = None
    contact_number: Optional[str] = None
    face_count: int = 0
    image_path: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
