from pydantic import BaseModel
from typing import List

class FaceEncodingRequest(BaseModel):
    """
    Request body containing a face encoding.
    """
    encoding: List[float]

class KnownFacesResponse(BaseModel):
    """
    Response for retrieving all known faces.
    """
    face_ids: List[int]
    face_encodings: List[List[float]]

class NewFaceResponse(BaseModel):
    """
    Response for creating a new face.
    """
    face_id: int
