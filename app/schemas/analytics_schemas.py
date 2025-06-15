from pydantic import BaseModel
from typing import List, Dict, Any

class FaceAppearanceData(BaseModel):
    face_id: int
    detection_time: str
    image_path: str

class FaceAppearanceDataResponse(BaseModel):
    data: List[FaceAppearanceData]

class FaceBatchDetailsResponse(BaseModel):
    details: Dict[int, Any]  # or create a more specific model

class FaceCoOccurrencesResponse(BaseModel):
    co_occurrence_matrix: List[List[float]]

class FaceTimeDistributionResponse(BaseModel):
    hour_distribution: Dict[int, int]
    day_distribution: Dict[str, int]
    total_appearances: int
