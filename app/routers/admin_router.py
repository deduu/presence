from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Optional
from app.crud.attendance import AsyncDatabaseHandler
from app.schemas.admin_schemas import (
    DeleteOldRecordsResponse,
    ExportDataFormat,
    ExportDataResponse
)

router = APIRouter()
db_handler = AsyncDatabaseHandler()

@router.delete("/cleanup", response_model=DeleteOldRecordsResponse)
async def delete_old_records(cutoff_date: str):
    """
    Delete records older than the specified cutoff_date (YYYY-MM-DD).
    """
    try:
        dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
        deleted_counts = await db_handler.delete_old_records(dt)
        return DeleteOldRecordsResponse(**deleted_counts)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD).")

@router.get("/export", response_model=ExportDataResponse)
async def export_data(format_type: ExportDataFormat = Query(ExportDataFormat.csv)):
    """
    Export face data in csv/json/excel.
    Returns a base64-encoded string of the file's binary content.
    """
    try:
        file_bytes = await db_handler.export_data(format_type=format_type.value)
        # Return bytes as base64 string so it can be downloaded.
        import base64
        encoded = base64.b64encode(file_bytes).decode("utf-8")

        return ExportDataResponse(
            file_content=encoded,
            file_format=format_type.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
