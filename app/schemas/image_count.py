from pydantic import BaseModel, field_serializer
from datetime import datetime
from typing import Optional
from app.schemas.image import ImageInDB


class ImageCountCreate(BaseModel):
    image_path: str
    face_count: int
    processed_time: datetime


class ImageCountUpdate(BaseModel):
    face_count: int
    processed_time: datetime


class ImageCountInDB(BaseModel):
    image_id: int
    face_count: int
    processed_time: datetime
    image: Optional[ImageInDB] = None

    @field_serializer("processed_time")
    def ser_dt(self, dt: datetime) -> str:
        return dt.isoformat()

    class Config:
        from_attributes = True
