# schemas/face.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class FaceBase(BaseModel):
    first_seen: datetime
    last_seen: datetime


class FaceCreate(FaceBase):
    face_encoding: bytes


class FaceInDB(FaceBase):
    face_id: int
    face_encoding: bytes

    class Config:
        orm_mode = True


class FaceOut(BaseModel):
    face_id: int
    first_seen: datetime
    last_seen: datetime
    person_id: Optional[int]
    person_name: Optional[str] = None  # joined via query
    thumbnail_url: Optional[str] = None
    person_image_url: Optional[str] = None

    class Config:
        orm_mode = True


# schemas/image_record.py
from pydantic import BaseModel
from datetime import datetime


class ImageRecordBase(BaseModel):
    image_path: str
    face_id: int
    detection_time: datetime


class ImageRecordCreate(ImageRecordBase):
    pass


class ImageRecordInDB(ImageRecordBase):
    record_id: int

    class Config:
        orm_mode = True


class ImageRecordOut(BaseModel):
    record_id: int
    face_id: int
    image_path: str
    detection_time: datetime
    person_name: str | None = None

    class Config:
        orm_mode = True


# schemas/image_count.py
from pydantic import BaseModel
from datetime import datetime


class ImageCountBase(BaseModel):
    image_path: str
    face_count: int
    processed_time: datetime


class ImageCountCreate(ImageCountBase):
    pass


class ImageCountInDB(ImageCountBase):
    image_id: int

    class Config:
        orm_mode = True


class ImageCountOut(BaseModel):
    image_id: int
    image_path: str
    face_count: int
    processed_time: datetime

    class Config:
        orm_mode = True
