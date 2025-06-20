from .base_service import BaseService
from app.db.models.attendance import Person

class PeopleService(BaseService):
    def __init__(self, db): super().__init__(db, Person)
