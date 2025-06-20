from pydantic import BaseModel
from datetime import date
from typing import Optional

class PersonBase(BaseModel):
    name: str
    date_of_birth: Optional[date] = None
    address: Optional[str] = None
    contact_number: Optional[str] = None

class PersonCreate(PersonBase): pass
class PersonUpdate(PersonBase): pass

class PersonOut(PersonBase):
    person_id: int
    class Config: orm_mode = True
