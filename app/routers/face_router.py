from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import numpy as np

# Import your AsyncDatabaseHandler
from app.crud.attendance import AsyncDatabaseHandler
# Pydantic schemas (for request/response)
from app.schemas.face_schemas import (
    FaceEncodingRequest,
    NewFaceResponse,
    KnownFacesResponse
)

router = APIRouter()
db_handler = AsyncDatabaseHandler()

@router.get("/known", response_model=KnownFacesResponse)
async def get_all_known_faces():
    """
    Retrieve all known faces and their encodings.
    For demonstration, returns face IDs and encodings as lists of floats.
    """
    known_face_ids, known_face_encodings = await db_handler.get_all_known_faces()
    # Convert each face encoding (np.ndarray) to a list of floats for serialization:
    encodings_as_lists = [enc.tolist() for enc in known_face_encodings]
    return KnownFacesResponse(
        face_ids=known_face_ids,
        face_encodings=encodings_as_lists
    )

@router.post("/", response_model=NewFaceResponse)
async def insert_new_face(request: FaceEncodingRequest):
    """
    Insert a new face with a given encoding.
    """
    # Convert the encoding (list of floats) to np.ndarray
    np_encoding = np.array(request.encoding, dtype=np.float64)
    current_time = datetime.utcnow()
    
    face_id = await db_handler.insert_new_face(np_encoding, current_time)
    if not face_id:
        raise HTTPException(status_code=500, detail="Failed to insert new face")
    
    return NewFaceResponse(face_id=face_id)

@router.patch("/{face_id}")
async def update_last_seen_face(face_id: int, request: FaceEncodingRequest = None):
    """
    Update last_seen for a face, optionally updating the face encoding as well.
    """
    current_time = datetime.utcnow()
    np_encoding = None
    
    if request and request.encoding:
        np_encoding = np.array(request.encoding, dtype=np.float64)
    
    await db_handler.update_last_seen(face_id, current_time, face_encoding=np_encoding)
    return {"message": f"Face {face_id} updated"}
