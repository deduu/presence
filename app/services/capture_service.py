import cv2
import time
import aiohttp
import asyncio
from datetime import datetime

class CCTVCaptureService:
    def __init__(self, rtsp_url: str, camera_id: str, interval: int = 5):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.interval = interval
        self.running = True

    async def capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"[{self.camera_id}] Cannot open stream")
            return

        while self.running:
            ret, frame = cap.read()
            if ret:
                filename = f"frame_{self.camera_id}_{datetime.utcnow().timestamp()}.jpg"
                cv2.imwrite(filename, frame)
                await self.send_to_backend(filename)

            await asyncio.sleep(self.interval)

        cap.release()

    async def send_to_backend(self, filepath):
        url = "http://localhost:8004/uploads/upload-preview"
        data = aiohttp.FormData()
        data.add_field("files", open(filepath, "rb"), filename=filepath)
        data.add_field("batch_tag", f"CCTV-{self.camera_id}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                print(f"[{self.camera_id}] Uploaded frame {filepath} → {resp.status}")
