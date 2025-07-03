# import uvicorn
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from typing import AsyncGenerator
# from fastapi import Depends
# import logging
# from fastapi.staticfiles import StaticFiles

# from sqlalchemy.ext.asyncio import AsyncSession

# from app.db.base import session_manager
# from app.api.v1.file_router import router as file_router
# from app.api.v1.face_router import router as face_router
# from app.api.v1.image_record_router import router as image_record_router
# from app.api.v1.image_count_router import router as image_count_router
# from app.api.v1.people_router import router as people_router
# from app.api.v1.dashboard_router import router as dashboard_router

# # Import file storage paths
# from app.utils.file_store import (
#     SERVER_IMAGE_STORAGE_ROOT, PUBLIC_IMAGE_URL_PREFIX,
#     TEMP_IMAGE_STORAGE_ROOT, PUBLIC_TEMP_IMAGE_URL_PREFIX,
#     SERVER_FACE_CROP_STORAGE_ROOT, PUBLIC_FACE_CROP_PREFIX,
#     SERVER_PERSON_IMAGE_ROOT, PUBLIC_PERSON_IMAGE_PREFIX,
# )
# # from app.api.v1.report_router import router as report_router

# from app.utils.logger import configure_logging
# configure_logging()


# logger = logging.getLogger(__name__)

# async def get_db_session():
#     """
#     Dependency that provides a SQLAlchemy AsyncSession.
#     """
#     async with session_manager.create_session() as session:
#         yield session


# app = FastAPI(
#     title="Face Recognition API",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
# )

# # Allow React dev server to talk to this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",  # Vite default
#         "http://localhost:3000",  # CRA default
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.mount(
#     PUBLIC_IMAGE_URL_PREFIX,
#     StaticFiles(directory=SERVER_IMAGE_STORAGE_ROOT),
#     name="static_images" # A name for the route, can be anything
# )

# # NEW: Mount static files for temporary preview images
# app.mount(
#     PUBLIC_TEMP_IMAGE_URL_PREFIX,
#     StaticFiles(directory=TEMP_IMAGE_STORAGE_ROOT),
#     name="static_temp_images"
# )

# app.mount(
#     PUBLIC_FACE_CROP_PREFIX,
#     StaticFiles(directory=str(SERVER_FACE_CROP_STORAGE_ROOT)),
#     name="face_crops"
# )

# app.mount(
#     PUBLIC_PERSON_IMAGE_PREFIX,
#     StaticFiles(directory=str(SERVER_PERSON_IMAGE_ROOT)),
#     name="person_images"
# )

# # Mount each router, passing the DB dependency
# app.include_router(file_router,      prefix="/uploads",   tags=["uploads"], dependencies=[Depends(get_db_session)])
# app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"], dependencies=[Depends(get_db_session)])
# app.include_router(people_router,    prefix="/people",    tags=["people"], dependencies=[Depends(get_db_session)])
# app.include_router(
#     face_router,
#     prefix="/faces",
#     tags=["faces"],
#     dependencies=[Depends(get_db_session)],
# )

# app.include_router(
#     image_record_router,
#     prefix="/image-records",
#     tags=["image-records"],
#     dependencies=[Depends(get_db_session)],
# )

# app.include_router(
#     image_count_router,
#     prefix="/image-counts",
#     tags=["image-counts"],
#     dependencies=[Depends(get_db_session)],
# )

# # app.include_router(
# #     report_router,
# #     prefix="/reports",
# #     tags=["reports"],
# #     dependencies=[Depends(get_db_session)],
# # )


# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8006,
#         log_level="info",
#         reload=True,           # remove in production
#     )
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.routing import APIRoute

from app.db.base import Base, session_manager, engine
from app.utils.logger import configure_logging
from app.utils.file_store import (
    SERVER_IMAGE_STORAGE_ROOT, PUBLIC_IMAGE_URL_PREFIX,
    TEMP_IMAGE_STORAGE_ROOT, PUBLIC_TEMP_IMAGE_URL_PREFIX,
    SERVER_FACE_CROP_STORAGE_ROOT, PUBLIC_FACE_CROP_PREFIX,
    SERVER_PERSON_IMAGE_ROOT, PUBLIC_PERSON_IMAGE_PREFIX,
)
from app.api.v1.file_router import router as file_router
from app.api.v1.face_router import router as face_router
from app.api.v1.image_record_router import router as image_record_router
from app.api.v1.image_count_router import router as image_count_router
from app.api.v1.people_router import router as people_router
from app.api.v1.dashboard_router import router as dashboard_router

configure_logging()
logger = logging.getLogger(__name__)

INTERNAL_IPS = {"127.0.0.1", "localhost", "::1"}


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
    logger.info("Database initialized.")
    yield
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
