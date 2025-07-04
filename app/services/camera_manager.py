import asyncio
import cv2
import json
import time
import uuid
import base64
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from app.schemas.camera_schemas import CameraConfig, CameraFrame
import logging
from app.services.ws_broadcaster_service import broadcast_text

logger = logging.getLogger(__name__)

# Shared state
active_connections = []
frame_buffer = {}
camera_configs = {}


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.running = False
        self.frame_tasks = {}

    async def add_camera(self, config: CameraConfig):
        """Add a new camera (web or CCTV)"""
        try:
            # determine the actual capture source
            if config.type == "web":
                # make sure it's an int index
                idx = int(config.source)
                # on Windows, force DirectShow backend
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(config.source)

            if not cap.isOpened():
                raise Exception(
                    f"Failed to open camera source: {config.source}")

            # Set camera properties
            width, height = config.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, config.fps)

            # Store
            self.cameras[config.camera_id] = {
                'capture': cap,
                'config': config,
                'active': True
            }
            camera_configs[config.camera_id] = config
            logger.info(f"Camera {config.camera_id} added ({config.type})")

        except Exception as e:
            logger.error(f"Error adding camera {config.camera_id}: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def remove_camera(self, camera_id: str):
        if camera := self.cameras.pop(camera_id, None):
            camera['capture'].release()
            camera_configs.pop(camera_id, None)
            logger.info(f"Camera {camera_id} removed")

    async def start_streaming(self):
        self.running = True
        for cid in self.cameras:
            self.frame_tasks[cid] = asyncio.create_task(
                self.stream_camera(cid))

    async def stop_streaming(self):
        self.running = False
        for t in self.frame_tasks.values():
            t.cancel()
        self.frame_tasks.clear()

    async def stream_camera(self, camera_id: str):
        """Continuously capture, process, encode, and broadcast frames."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return

        cap = camera['capture']
        config = camera['config']
        logger.info(f"[{camera_id}] entering stream loop (fps={config.fps})")

        while self.running and camera['active']:
            # 1) Grab a frame
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            # 2) Process (e.g. resize + timestamp overlay)
            processed = await self.process_frame(frame, camera_id)

            # 3) Encode to JPEG in memory
            success, buf = cv2.imencode(".jpg", processed)
            if not success:
                await asyncio.sleep(0.1)
                continue
            jpg_bytes = buf.tobytes()

            # 4) Build a JSON payload
            b64 = base64.b64encode(jpg_bytes).decode("ascii")
            payload = {
                "camera_id": camera_id,
                "timestamp": datetime.utcnow().isoformat(),
                "frame": b64,
            }
            text = json.dumps(payload)

            # 5) Broadcast to all connected WS clients
            await broadcast_text(text)

            # 6) Wait to match the desired FPS
            await asyncio.sleep(1.0 / config.fps)

    async def process_frame(self, frame, camera_id: str):
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{camera_id} - {ts}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame


camera_manager = CameraManager()
