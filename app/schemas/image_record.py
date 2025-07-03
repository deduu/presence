from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_serializer
from app.schemas.image import ImageInDB


class ImageRecordBase(BaseModel):
    image_id: int
    image_url: str
    face_id: int
    detection_time: datetime


class ImageRecordCreate(ImageRecordBase):
    pass


class ImageRecordInDB(ImageRecordBase):
    record_id: int
    face_location: Optional[str] = None
    image: Optional[ImageInDB]
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    batch_tag: Optional[str] = None

    @field_serializer("detection_time")
    def ser_dt(self, dt: datetime) -> str:
        return dt.isoformat()

    class Config:
        from_attributes = True


class ImageRecordOut(BaseModel):
    record_id: int
    image_id: int
    image_url: str
    image_path: str
    face_id: int
    face_location: List[float]
    image_width: int
    image_height: int
    detection_time: datetime
    person_name: Optional[str] = None
    batch_tag: Optional[str] = None

    @field_serializer("detection_time")
    def ser_dt(self, dt: datetime) -> str:
        return dt.isoformat()

    class Config:
        from_attributes = True
