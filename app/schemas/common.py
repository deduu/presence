
# schemas/person.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PersonBase(BaseModel):
    name: str
    date_of_birth: Optional[datetime] = None
    address: Optional[str] = None
    contact_number: Optional[str] = None

class PersonCreate(PersonBase):
    pass

class PersonRead(PersonBase):
    person_id: int

    class Config:
        orm_mode = True
