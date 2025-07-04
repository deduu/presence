# app/camera/camera_simulator.py
import cv2
import asyncio
import os
from uuid import uuid4
from datetime import datetime
from app.services.ws_broadcaster_service import broadcast_frame

# ─── cache last frame for every camera ─────────────────────────────
latest: dict[str, dict] = {}


async def simulate_camera(camera_id: str, video_path: str):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    print(f"[{camera_id}] opened? {cap.isOpened()} path={os.path.abspath(video_path)}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Reached end, rewinding...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        filename = f"frames/{camera_id}_{uuid4().hex[:6]}.jpg"
        os.makedirs("frames", exist_ok=True)
        cv2.imwrite(filename, frame)

       # update cache ───────────────────────────────────────────────
        latest[camera_id] = {
            "camera_id": camera_id,
            "image_url": f"/frames/{os.path.basename(filename)}",
            "timestamp": datetime.utcnow().isoformat(),
        }

        await asyncio.sleep(1)           # ← still 1 FPS capture; no send

    cap.release()


async def start_camera_feeds():
    await asyncio.gather(
        simulate_camera("Camera-1", "sample_video_1.mp4"),
        simulate_camera("Camera-2", "sample_video_2.mp4"),
    )
