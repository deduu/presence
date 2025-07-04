# app/services/ws_broadcaster.py
import asyncio
import json
from typing import List
from app.schemas.camera_schemas import CameraFrame

# each connected client gets its own asyncio.Queue
active_queues: List[asyncio.Queue] = []


async def broadcast_frame(frame_data: CameraFrame):
    """Enqueue a new frame for every connected WebSocket."""
    msg = json.dumps(frame_data.dict())
    # iterate over a copy in case someone disconnects mid‐loop
    for q in list(active_queues):
        try:
            q.put_nowait(frame_data)
        except asyncio.QueueFull:
            # drop if queue is clogged
            pass
        except Exception:
            # if the queue is invalid, remove it
            active_queues.remove(q)


async def broadcast_bytes(camera_id: str, jpg_bytes: bytes):
    """Enqueue a new frame (camera_id + raw JPEG) for every WS client."""
    for q in list(active_queues):
        try:
            q.put_nowait((camera_id, jpg_bytes))
        except asyncio.QueueFull:
            # drop if the client is too slow
            pass
        except Exception:
            active_queues.remove(q)


async def broadcast_text(text: str):
    """Enqueue a new frame (camera_id + raw JPEG) for every WS client."""
    for q in list(active_queues):
        try:
            q.put_nowait(text)
        except asyncio.QueueFull:
            # drop if the client is too slow
            pass
