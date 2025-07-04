from app.api.v1.dashboard_router import router as dashboard_router
from app.api.v1.people_router import router as people_router
from app.api.v1.image_count_router import router as image_count_router
from app.api.v1.image_record_router import router as image_record_router
from app.api.v1.face_router import router as face_router
from app.api.v1.file_router import router as file_router
from app.api.v1.ws_router import router as ws_router
from app.api.v1.camera_router import router as camera_router
from app.utils.file_store import (
    SERVER_IMAGE_STORAGE_ROOT, PUBLIC_IMAGE_URL_PREFIX,
    TEMP_IMAGE_STORAGE_ROOT, PUBLIC_TEMP_IMAGE_URL_PREFIX,
    SERVER_FACE_CROP_STORAGE_ROOT, PUBLIC_FACE_CROP_PREFIX,
    SERVER_PERSON_IMAGE_ROOT, PUBLIC_PERSON_IMAGE_PREFIX,
)
from app.utils.logger import configure_logging
from app.db.base import Base, session_manager, engine
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import logging
import asyncio
import os
from app.camera.camera_simulator import start_camera_feeds
from app.services.camera_manager import camera_manager

configure_logging()
logger = logging.getLogger(__name__)

INTERNAL_IPS = {"127.0.0.1", "localhost", "::1"}

# Ensure static/images exists at import time, before mounting
base_dir = os.getcwd()
images_dir = os.path.join(base_dir, "static", "images")
logger.info(f"Ensuring {images_dir} exists…")
os.makedirs(images_dir, exist_ok=True)


class RestrictRootAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        client_ip = request.client.host
        if path == "/" and client_ip not in INTERNAL_IPS:
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("CCTV System starting up...")

    # logger.info("Starting simulated camera feeds...")
    # asyncio.create_task(start_camera_feeds())  # ✅ must be before yield

    yield  # app is now live
    logger.info("CCTV System shutting down...")
    await camera_manager.stop_streaming()
    for cam_data in camera_manager.cameras.values():
        cam_data['capture'].release()

    logger.info("Cleanup complete")
    logger.info("Shutting down...")
    await session_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Recognition API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(RestrictRootAccessMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount(PUBLIC_IMAGE_URL_PREFIX, StaticFiles(
        directory=SERVER_IMAGE_STORAGE_ROOT), name="static_images")
    app.mount(PUBLIC_TEMP_IMAGE_URL_PREFIX, StaticFiles(
        directory=TEMP_IMAGE_STORAGE_ROOT), name="static_temp_images")
    app.mount(PUBLIC_FACE_CROP_PREFIX, StaticFiles(
        directory=SERVER_FACE_CROP_STORAGE_ROOT), name="face_crops")
    app.mount(PUBLIC_PERSON_IMAGE_PREFIX, StaticFiles(
        directory=SERVER_PERSON_IMAGE_ROOT), name="person_images")

    # Serve static images from /frames
    app.mount("/frames", StaticFiles(directory="frames"), name="frames")
    app.mount("/images", StaticFiles(directory="static/images"), name="images")

    # Add routers
    app.include_router(file_router, prefix="/uploads", tags=["uploads"])
    app.include_router(
        dashboard_router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(people_router, prefix="/people", tags=["people"])
    app.include_router(face_router, prefix="/faces", tags=["faces"])
    app.include_router(image_record_router,
                       prefix="/image-records", tags=["image-records"])
    app.include_router(image_count_router,
                       prefix="/image-counts", tags=["image-counts"])
    app.include_router(camera_router, prefix="/cameras", tags=["cameras"])
    app.include_router(ws_router, tags=["ws"])

    return app


app = create_app()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    client_ip = request.client.host
    if client_ip in INTERNAL_IPS:
        return "<h1>Welcome</h1>"
    return HTMLResponse("<h1>Not allowed</h1>", status_code=403)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8006, reload=True)
