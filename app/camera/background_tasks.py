import asyncio
from app.camera.camera_simulator import simulate_camera


async def start_camera_feeds():
    await asyncio.gather(
        simulate_camera("Camera-1", "sample_video_1.mp4"),
        simulate_camera("Camera-2", "sample_video_2.mp4")
    )
