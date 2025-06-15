from fastapi import FastAPI, Depends
from app.crud.attendance import AsyncDatabaseHandler
import asyncio

app = FastAPI()
db_handler = AsyncDatabaseHandler()

@app.get("/get_faces/")
async def get_all_faces():
    known_face_ids, known_face_encodings = await db_handler.get_all_known_faces()
    return {"face_ids": known_face_ids, "encodings": [enc.tolist() for enc in known_face_encodings]}

@app.get("/get_face_details/{face_id}")
async def get_face_details(face_id: int):
    return await db_handler.get_face_details_batch([face_id])

@app.get("/get_face_co_occurrences/")
async def get_face_co_occurrences(face_ids: str):
    face_ids_list = [int(x) for x in face_ids.split(",")]
    co_matrix = await db_handler.get_face_co_occurrences(face_ids_list)
    return {"co_occurrence_matrix": co_matrix.tolist()}
