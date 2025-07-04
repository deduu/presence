# app/api/v1/ws_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from app.services.ws_broadcaster_service import active_queues

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    active_queues.append(queue)
    logger.info(
        f"[WS] Client connected -> total queues = {len(active_queues)}")

    # # Send any buffered frames immediately
    # from app.services.camera_manager import frame_buffer
    # for frame in frame_buffer.values():
    #     await ws.send_text(json.dumps(frame.dict()))

    # Sender task: pulls from THIS client's queue
    async def sender():
        try:
            while True:
                message_text: str = await queue.get()
                await ws.send_text(message_text)
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(f"[WS] sender error: {e}")

    send_task = asyncio.create_task(sender())

    try:
        # just keep the socket alive
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        active_queues.remove(queue)
        logger.info(
            f"[WS] Client disconnected -> total queues = {len(active_queues)}")
