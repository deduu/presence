from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.camera_schemas import CameraConfig
from app.services.camera_manager import camera_manager, camera_configs

router = APIRouter()


@router.post("/add")
async def add_camera(config: CameraConfig):
    await camera_manager.add_camera(config)
    return {"message": f"Camera {config.camera_id} added"}


@router.delete("/{camera_id}")
async def remove_camera(camera_id: str):
    await camera_manager.remove_camera(camera_id)
    return {"message": f"Camera {camera_id} removed"}


@router.get("/", response_model=List[CameraConfig])
async def list_cameras():
    return list(camera_configs.values())


@router.get("/{camera_id}/status")
async def camera_status(camera_id: str):
    if camera_id not in camera_configs:
        raise HTTPException(404, "Not found")
    return {"active": camera_id in camera_manager.cameras, "config": camera_configs[camera_id]}


@router.post("/start")
async def start_stream():
    await camera_manager.start_streaming()
    return {"message": "Streaming started"}


@router.post("/stop")
async def stop_stream():
    await camera_manager.stop_streaming()
    return {"message": "Streaming stopped"}

# @router.get("/metrics")
# async def get_metrics():
#     return {
#         "total_people": len({d.person_name for d in detection_results}),
#         "total_faces": len(detection_results),
#         "total_records": len(detection_results),
#         "active_cameras": len(camera_manager.cameras)
#     }
