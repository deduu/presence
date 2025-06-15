from pydantic import BaseModel
from enum import Enum

class DeleteOldRecordsResponse(BaseModel):
    image_records: int
    image_counts: int
    faces: int

class ExportDataFormat(str, Enum):
    csv = "csv"
    json = "json"
    excel = "excel"

class ExportDataResponse(BaseModel):
    file_content: str
    file_format: str
