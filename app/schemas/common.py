# schemas/face.py

from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from typing import Optional, List


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
    first_seen: Optional[datetime]  # ✅ Allow None
    last_seen: Optional[datetime]   # ✅ Allow None
    person_id: Optional[int]
    person_name: Optional[str] = None  # joined via query
    thumbnail_url: Optional[str] = None
    person_image_url: Optional[str] = None

    class Config:
        orm_mode = True


# schemas/image_record.py


class ImageRecordBase(BaseModel):
    image_path: str
    image_url: str
    face_id: int
    detection_time: datetime


class ImageRecordCreate(ImageRecordBase):
    pass


class ImageRecordInDB(ImageRecordBase):
    record_id: int

    @field_serializer("detection_time")
    def ser_dt(self, dt: datetime) -> str:      # <- converts for JSON
        return dt.isoformat()

    class Config:
        orm_mode = True


class ImageRecordOut(BaseModel):
    record_id: int
    face_id: int
    image_path: str
    image_url: str

    detection_time: datetime
    face_location: List[float]
    image_width: int
    image_height: int
    person_name: str | None = None

    @field_serializer("detection_time")
    def ser_dt(self, dt: datetime) -> str:      # <- converts for JSON
        return dt.isoformat()

    class Config:
        orm_mode = True


# schemas/image_count.py


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
