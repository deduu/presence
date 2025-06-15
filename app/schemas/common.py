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
