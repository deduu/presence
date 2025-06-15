from fastapi import FastAPI
from app.routers.face_router import router as face_router
from app.routers.image_router import router as image_router
from app.routers.analytics_router import router as analytics_router
from app.routers.admin_router import router as admin_router

# Create the FastAPI application
app = FastAPI(title="Face Recognition API", version="1.0.0")

# Include routers
app.include_router(face_router, prefix="/api/faces", tags=["Faces"])
app.include_router(image_router, prefix="/api/images", tags=["Images"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

# Optional startup/shutdown events
@app.on_event("startup")
async def on_startup():
    """
    Perform any startup logic here, e.g.:
    - Checking DB connectivity
    - Seeding data, etc.
    """
    pass

@app.on_event("shutdown")
async def on_shutdown():
    """
    Perform any teardown logic here, if needed.
    """
    pass
