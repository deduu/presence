from pydantic import BaseModel
from typing import Tuple, List


from pydantic import BaseModel, validator
from typing import Tuple, Literal, Union


class CameraConfig(BaseModel):
    camera_id: str
    name: str
    # source can be device index (int) or URL/file path (str)
    source: Union[int, str]
    # type indicates how to interpret source
    type: Literal['web', 'cctv'] = 'web'
    enabled: bool = True
    fps: int = 30
    resolution: Tuple[int, int] = (640, 480)

    @validator('source', pre=True)
    def cast_web_index(cls, v, values):
        # if type is web and source is numeric string, cast to int
        if values.get('type') == 'web' and isinstance(v, str) and v.isdigit():
            return int(v)
        return v


class CameraFrame(BaseModel):
    camera_id: str
    image_url: str
    timestamp: str
    frame_id: str


class DetectionResult(BaseModel):
    face_id: str
    person_name: str
    image_path: str
    detection_time: str
    confidence: float
    camera_id: str
