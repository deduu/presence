import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator
from fastapi import Depends


from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import session_manager
from app.routers.face_router import router as face_router
from app.api.v1.image_record_router import router as image_record_router
from app.api.v1.image_count_router import router as image_count_router
# from app.api.v1.report_router import router as report_router

async def get_db_session():
    """
    Dependency that provides a SQLAlchemy AsyncSession.
    """
    async with session_manager.create_session() as session:
        yield session



app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow React dev server to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # CRA default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount each router, passing the DB dependency
app.include_router(
    face_router,
    prefix="/faces",
    tags=["faces"],
    dependencies=[Depends(get_db_session)],
)

app.include_router(
    image_record_router,
    prefix="/image-records",
    tags=["image-records"],
    dependencies=[Depends(get_db_session)],
)

app.include_router(
    image_count_router,
    prefix="/image-counts",
    tags=["image-counts"],
    dependencies=[Depends(get_db_session)],
)

# app.include_router(
#     report_router,
#     prefix="/reports",
#     tags=["reports"],
#     dependencies=[Depends(get_db_session)],
# )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8006,
        log_level="info",
        reload=True,           # remove in production
    )
