# app/routers/dashboard.py
from datetime import datetime, timedelta, date

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import session_manager
from app.db.models.attendance import Person, Face, ImageRecord

router = APIRouter()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _scalar(session: AsyncSession, stmt):
    """Return a single scalar result ( COUNT(*) etc. )"""
    result = await session.execute(stmt)
    return result.scalar_one()

async def get_db_session():
    async with session_manager.create_session() as session:
        yield session
# --------------------------------------------------------------------------- #
# 1)  /dashboard/metrics
# --------------------------------------------------------------------------- #
@router.get("/metrics")
async def metrics(session: AsyncSession = Depends(get_db_session)):
    total_people  = await _scalar(session, select(func.count(Person.person_id)))
    total_faces   = await _scalar(session, select(func.count(Face.face_id)))
    total_records = await _scalar(session, select(func.count(ImageRecord.record_id)))
    return {
        "total_people":  total_people,
        "total_faces":   total_faces,
        "total_records": total_records,
    }


# --------------------------------------------------------------------------- #
# 2)  /dashboard/recent   → latest 10 detections with person name if any
# --------------------------------------------------------------------------- #
@router.get("/recent")
async def recent_detections(session: AsyncSession = Depends(get_db_session), limit: int = 10):
    stmt = (
        select(
            ImageRecord.record_id,
            ImageRecord.image_path,
            ImageRecord.detection_time,
            Face.face_id,
            func.coalesce(Person.name, "Anonymous").label("person_name"),
        )
        .join(Face, ImageRecord.face_id == Face.face_id)
        .join(Person, Face.person_id == Person.person_id, isouter=True)
        .order_by(ImageRecord.detection_time.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = result.all()
    return [
        {
            "face_id":        r.face_id,
            "person_name":    r.person_name,
            "image_path":     r.image_path,
            "detection_time": r.detection_time,
        }
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# 3)  /dashboard/charts
#     - faces per day (last 7 days)
#     - top 5 most-seen people
# --------------------------------------------------------------------------- #
@router.get("/charts")
async def charts(session: AsyncSession = Depends(get_db_session)):
    # ---- Faces detected per day (last 7 days) ------------------------------ #
    today: date   = datetime.utcnow().date()
    start: date   = today - timedelta(days=6)   # inclusive 7-day window

    daily_stmt = (
        select(
            cast(ImageRecord.detection_time, Date).label("d"),
            func.count().label("cnt"),
        )
        .where(ImageRecord.detection_time >= start)
        .group_by("d")
        .order_by("d")
    )
    daily_rows = await session.execute(daily_stmt)
    daily_map  = {row.d: row.cnt for row in daily_rows}   # dict(date → count)

    days   = []
    counts = []
    for i in range(7):
        d = start + timedelta(days=i)
        days.append(d.isoformat())
        counts.append(daily_map.get(d, 0))

    # ---- Top 5 most-seen people  ------------------------------------------ #
    top_stmt = (
        select(
            func.coalesce(Person.name, "Anonymous").label("person_name"),
            func.count().label("total"),
        )
        .join(Face, Face.person_id == Person.person_id, isouter=True)
        .join(ImageRecord, ImageRecord.face_id == Face.face_id)
        .group_by("person_name")
        .order_by(func.count().desc())
        .limit(5)
    )
    top_rows = await session.execute(top_stmt)
    top_people = [{"name": r.person_name, "count": r.total} for r in top_rows]

    return {
        "days":      days,          # ['2025-06-14', ... '2025-06-20']
        "counts":    counts,        # [3,2,5, ...]
        "topPeople": top_people,    # [{name:'Alice',count:18}, ...]
    }
