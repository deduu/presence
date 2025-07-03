from pydantic import BaseModel
from typing import Optional


class ImageInDB(BaseModel):
    image_id: int
    image_path: str

    class Config:
        from_attributes = True
