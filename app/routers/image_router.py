from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.crud.attendance import AsyncDatabaseHandler
from app.schemas.image_schemas import (
    ImageRecordRequest,
    ImageRecordResponse,
    ImageCountRequest,
    ImageCountResponse
)

router = APIRouter()
db_handler = AsyncDatabaseHandler()

@router.post("/record", response_model=ImageRecordResponse)
async def insert_image_record(request: ImageRecordRequest):
    """
    Insert a new image record into the database.
    """
    try:
        detection_time = datetime.utcnow()
        await db_handler.insert_image_record(
            image_path=request.image_path,
            face_id=request.face_id,
            detection_time=detection_time
        )
        return ImageRecordResponse(
            message=f"Image record inserted for face_id: {request.face_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/count", response_model=ImageCountResponse)
async def insert_or_update_image_count(request: ImageCountRequest):
    """
    Insert or update the face count for an image.
    """
    try:
        processed_time = datetime.utcnow()
        await db_handler.insert_or_update_image_count(
            image_path=request.image_path,
            face_count=request.face_count,
            processed_time=processed_time
        )
        return ImageCountResponse(
            message=f"Image count updated for {request.image_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
