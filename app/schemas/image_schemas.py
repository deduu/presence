from pydantic import BaseModel

class ImageRecordRequest(BaseModel):
    image_path: str
    face_id: int

class ImageRecordResponse(BaseModel):
    message: str

class ImageCountRequest(BaseModel):
    image_path: str
    face_count: int

class ImageCountResponse(BaseModel):
    message: str
