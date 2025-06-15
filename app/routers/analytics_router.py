from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import numpy as np

from app.crud.attendance import AsyncDatabaseHandler
from app.schemas.analytics_schemas import (
    FaceAppearanceDataResponse,
    FaceBatchDetailsResponse,
    FaceCoOccurrencesResponse,
    FaceTimeDistributionResponse
)

router = APIRouter()
db_handler = AsyncDatabaseHandler()

@router.get("/face-appearance", response_model=FaceAppearanceDataResponse)
async def get_face_appearance_data():
    """
    Retrieve times and image paths where faces appeared (for visualization).
    """
    data = await db_handler.get_face_appearance_data()
    return FaceAppearanceDataResponse(data=data)

@router.post("/face-details-batch", response_model=FaceBatchDetailsResponse)
async def get_face_details_batch(face_ids: List[int]):
    """
    Retrieve details for multiple face IDs (appearance count, unique days, etc.).
    """
    details = await db_handler.get_face_details_batch(face_ids)
    return FaceBatchDetailsResponse(details=details)

@router.get("/image-count")
async def get_image_count():
    """
    Get the total number of unique images.
    """
    count = await db_handler.get_image_count()
    return {"count": count}

@router.post("/co-occurrences", response_model=FaceCoOccurrencesResponse)
async def get_face_co_occurrences(face_ids: List[int]):
    """
    Generate a co-occurrence matrix for how often a list of faces appear together.
    Returns a 2D list (matrix).
    """
    matrix = await db_handler.get_face_co_occurrences(face_ids)
    # Convert numpy array to list of lists
    matrix_list = matrix.tolist()
    return FaceCoOccurrencesResponse(co_occurrence_matrix=matrix_list)

@router.get("/time-distribution", response_model=FaceTimeDistributionResponse)
async def get_face_time_distribution(face_id: Optional[int] = None):
    """
    Get time distribution (by hour/day) for all faces or a single face if specified.
    """
    distribution = await db_handler.get_face_time_distribution(face_id=face_id)
    return FaceTimeDistributionResponse(**distribution)
