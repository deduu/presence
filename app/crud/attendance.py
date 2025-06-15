# database.py

import logging
from typing import List, Tuple, Optional
from sqlalchemy.future import select
import numpy as np

from sqlalchemy import insert, update
import asyncio
from app.db.base import session_manager
from app.db.models.attendance import Face, ImageRecord, ImageCount


logger = logging.getLogger(__name__)

class AsyncDatabaseHandler:
    def __init__(self):
        self.session_manager = session_manager  # Use the session manager you provided

    async def connect(self):
        # Connection is managed by session_manager
        pass

    async def close(self):
        await self.session_manager.close()

    async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
        """
        Returns two lists: known_face_ids and known_face_encodings
        """
        try:
            async with self.session_manager.create_session() as session:
                result = await session.execute(select(Face))
                faces = result.scalars().all()
                known_face_ids = [face.face_id for face in faces]
                known_face_encodings = [np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces]
                return known_face_ids, known_face_encodings
        except Exception as e:
            logger.error(f"Error retrieving known faces: {e}")
            return [], []

    async def insert_new_face(self, face_encoding: np.ndarray, current_time) -> Optional[int]:
        """
        Inserts a new face into the database and returns the face_id
        """
        try:
            face_encoding_bytes = face_encoding.tobytes()
            new_face = Face(
                face_encoding=face_encoding_bytes,
                first_seen=current_time,
                last_seen=current_time
            )
            async with self.session_manager.create_session() as session:
                session.add(new_face)
                await session.commit()
                await session.refresh(new_face)
                logger.info(f"Inserted new face with ID {new_face.face_id}")
                return new_face.face_id
        except Exception as e:
            logger.error(f"Error inserting new face: {e}")
            return None

    async def update_last_seen(self, face_id: int, current_time, face_encoding: Optional[np.ndarray] = None):
        """
        Updates the last_seen time (and optionally the face_encoding) of a face
        """
        try:
            async with self.session_manager.create_session() as session:
                stmt = select(Face).where(Face.face_id == face_id)
                result = await session.execute(stmt)
                face = result.scalar_one_or_none()
                if face:
                    face.last_seen = current_time
                    if face_encoding is not None:
                        face.face_encoding = face_encoding.tobytes()
                    await session.commit()
                    logger.info(f"Updated last_seen for face ID {face_id}")
                else:
                    logger.error(f"No face found with ID {face_id}")
        except Exception as e:
            logger.error(f"Error updating last_seen: {e}")

    async def insert_image_record(self, image_path: str, face_id: int, detection_time):
        """
        Inserts a new image record into the database
        """
        try:
            image_record = ImageRecord(
                image_path=image_path,
                face_id=face_id,
                detection_time=detection_time
            )
            async with self.session_manager.create_session() as session:
                session.add(image_record)
                await session.commit()
                logger.info(f"Inserted image record for face ID {face_id} and image {image_path}")
        except Exception as e:
            logger.error(f"Error inserting image record: {e}")

    async def insert_or_update_image_count(self, image_path: str, face_count: int, processed_time):
        """
        Inserts or updates the image count for an image
        """
        try:
            async with self.session_manager.create_session() as session:
                stmt = select(ImageCount).where(ImageCount.image_path == image_path)
                result = await session.execute(stmt)
                image_count = result.scalar_one_or_none()
                if image_count:
                    image_count.face_count = face_count
                    image_count.processed_time = processed_time
                    logger.info(f"Updated image count for {image_path}")
                else:
                    image_count = ImageCount(
                        image_path=image_path,
                        face_count=face_count,
                        processed_time=processed_time
                    )
                    session.add(image_count)
                    logger.info(f"Inserted image count for {image_path}")
                await session.commit()
        except Exception as e:
            logger.error(f"Error inserting/updating image count: {e}")

    # Additional CRUD methods can be added here as needed
    # Additional methods for AsyncDatabaseHandler in database.py

    async def get_face_appearance_data(self):
        """
        Returns data about face appearances for visualization
        
        Returns:
        list: A list of dictionaries with face_id, detection_time, and image_path
        """
        try:
            async with self.session_manager.create_session() as session:
                # Query to get all face appearances with timestamps
                stmt = select(
                    ImageRecord.face_id,
                    ImageRecord.detection_time,
                    ImageRecord.image_path
                ).order_by(ImageRecord.detection_time)
                
                result = await session.execute(stmt)
                records = result.all()
                
                # Convert to list of dictionaries
                face_data = [
                    {
                        "face_id": record.face_id,
                        "detection_time": record.detection_time,
                        "image_path": record.image_path
                    }
                    for record in records
                ]
                
                return face_data
        except Exception as e:
            logger.error(f"Error retrieving face appearance data: {e}")
            return []
        
    async def get_face_details_batch(self, face_ids):
        """
        Get details for multiple face IDs in a single query
        
        Parameters:
        face_ids (list): The IDs of the faces to get details for
        
        Returns:
        dict: A dictionary mapping face_ids to their details
        """
        try:
            result_dict = {}
            
            async with self.session_manager.create_session() as session:
                # Get face information for all face IDs at once
                face_stmt = select(Face).where(Face.face_id.in_(face_ids))
                face_result = await session.execute(face_stmt)
                faces = face_result.scalars().all()
                
                # Create a mapping of face_id to face object
                face_map = {face.face_id: face for face in faces}
                
                # Get all appearances for these faces in one query
                count_stmt = select(ImageRecord).where(ImageRecord.face_id.in_(face_ids))
                count_result = await session.execute(count_stmt)
                all_appearances = count_result.scalars().all()
                
                # Group appearances by face_id
                appearances_by_face = {}
                for appearance in all_appearances:
                    face_id = appearance.face_id
                    if face_id not in appearances_by_face:
                        appearances_by_face[face_id] = []
                    appearances_by_face[face_id].append(appearance)
                
                # Build the result dictionary
                for face_id in face_ids:
                    if face_id in face_map:
                        face = face_map[face_id]
                        appearances = appearances_by_face.get(face_id, [])
                        
                        # Get unique days this face appeared on
                        unique_days = set()
                        for appearance in appearances:
                            unique_days.add(appearance.detection_time.date())
                        
                        # Add face details to result dictionary
                        result_dict[face_id] = {
                            "face_id": face.face_id,
                            "first_seen": face.first_seen,
                            "last_seen": face.last_seen,
                            "appearance_count": len(appearances),
                            "unique_days": len(unique_days)
                        }
                
                return result_dict
        except Exception as e:
            logger.error(f"Error retrieving face details batch: {e}")
            return {}
    # async def get_face_details(self, face_id):
    #     """
    #     Get details for a specific face ID
        
    #     Parameters:
    #     face_id (int): The ID of the face to get details for
        
    #     Returns:
    #     dict: A dictionary containing details about the face
    #     """
    #     try:
    #         async with self.session_manager.create_session() as session:
    #             # Get face information
    #             face_stmt = select(Face).where(Face.face_id == face_id)
    #             face_result = await session.execute(face_stmt)
    #             face = face_result.scalar_one_or_none()
                
    #             if not face:
    #                 logger.warning(f"No face found with ID {face_id}")
    #                 return None
                
    #             # Count appearances
    #             count_stmt = select(ImageRecord).where(ImageRecord.face_id == face_id)
    #             count_result = await session.execute(count_stmt)
    #             appearances = count_result.scalars().all()
                
    #             # Get unique days this face appeared on
    #             unique_days = set()
    #             for appearance in appearances:
    #                 unique_days.add(appearance.detection_time.date())
                
    #             # Return face details dictionary
    #             return {
    #                 "face_id": face.face_id,
    #                 "first_seen": face.first_seen,
    #                 "last_seen": face.last_seen,
    #                 "appearance_count": len(appearances),
    #                 "unique_days": len(unique_days)
    #             }
    #     except Exception as e:
    #         logger.error(f"Error retrieving face details: {e}")
    #         return None

    async def get_image_count(self):
        """
        Get the total number of unique images in the database
        
        Returns:
        int: The number of unique images
        """
        try:
            async with self.session_manager.create_session() as session:
                stmt = select(ImageCount)
                result = await session.execute(stmt)
                images = result.scalars().all()
                return len(images)
        except Exception as e:
            logger.error(f"Error retrieving image count: {e}")
            return 0

    async def get_face_co_occurrences(self, face_ids):
        """
        Calculate a co-occurrence matrix showing how often selected faces appear together
        
        Parameters:
        face_ids (list): List of face IDs to analyze
        
        Returns:
        numpy.ndarray: A square matrix where each cell [i,j] represents how often face_i and face_j 
                    appear in the same image
        """
        try:
            async with self.session_manager.create_session() as session:
                # Initialize co-occurrence matrix
                n = len(face_ids)
                co_occurrence = np.zeros((n, n))
                
                # For each pair of faces, count images where both appear
                for i, face_i in enumerate(face_ids):
                    # Get all images where face_i appears
                    i_stmt = select(ImageRecord.image_path).where(ImageRecord.face_id == face_i)
                    i_result = await session.execute(i_stmt)
                    images_with_i = set([record[0] for record in i_result])
                    
                    # Count occurrence of face_i (diagonal value)
                    co_occurrence[i, i] = len(images_with_i)
                    
                    # For each other face, find overlap
                    for j in range(i+1, n):
                        face_j = face_ids[j]
                        
                        j_stmt = select(ImageRecord.image_path).where(ImageRecord.face_id == face_j)
                        j_result = await session.execute(j_stmt)
                        images_with_j = set([record[0] for record in j_result])
                        
                        # Intersection is images where both faces appear
                        intersection = images_with_i.intersection(images_with_j)
                        co_occurrence[i, j] = len(intersection)
                        co_occurrence[j, i] = len(intersection)  # Matrix is symmetric
                
                # Normalize to get proportions
                for i in range(n):
                    if co_occurrence[i, i] > 0:
                        for j in range(n):
                            co_occurrence[i, j] /= max(co_occurrence[i, i], 1)
                            
                return co_occurrence
        except Exception as e:
            logger.error(f"Error calculating face co-occurrences: {e}")
            return np.zeros((len(face_ids), len(face_ids)))

    async def get_face_time_distribution(self, face_id=None):
        """
        Get time distribution data for faces - when faces appear by hour/day
        
        Parameters:
        face_id (int, optional): If provided, only get data for this face
        
        Returns:
        dict: Dictionary containing time distribution data
        """
        try:
            async with self.session_manager.create_session() as session:
                if face_id:
                    # Query for a specific face
                    stmt = select(ImageRecord.detection_time).where(ImageRecord.face_id == face_id)
                else:
                    # Query for all faces
                    stmt = select(ImageRecord.detection_time)
                    
                result = await session.execute(stmt)
                timestamps = [record[0] for record in result]
                
                # Calculate distributions
                hours = [ts.hour for ts in timestamps]
                days = [ts.strftime('%A') for ts in timestamps]  # Day name
                
                # Count occurrences by hour and day
                hour_counts = {}
                for hour in range(24):
                    hour_counts[hour] = hours.count(hour)
                    
                day_counts = {}
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    day_counts[day] = days.count(day)
                    
                return {
                    'hour_distribution': hour_counts,
                    'day_distribution': day_counts,
                    'total_appearances': len(timestamps)
                }
        except Exception as e:
            logger.error(f"Error retrieving face time distribution: {e}")
            return {
                'hour_distribution': {},
                'day_distribution': {},
                'total_appearances': 0
            }

    async def delete_old_records(self, cutoff_date):
        """
        Delete records older than the cutoff date
        
        Parameters:
        cutoff_date (datetime): Date before which records should be deleted
        
        Returns:
        dict: Dictionary with counts of deleted records
        """
        try:
            deleted_counts = {
                'image_records': 0,
                'image_counts': 0,
                'faces': 0
            }
            
            async with self.session_manager.create_session() as session:
                # Delete old image records
                ir_stmt = select(ImageRecord).where(ImageRecord.detection_time < cutoff_date)
                ir_result = await session.execute(ir_stmt)
                old_ir = ir_result.scalars().all()
                
                for record in old_ir:
                    await session.delete(record)
                    deleted_counts['image_records'] += 1
                    
                # Delete old image counts
                ic_stmt = select(ImageCount).where(ImageCount.processed_time < cutoff_date)
                ic_result = await session.execute(ic_stmt)
                old_ic = ic_result.scalars().all()
                
                for count in old_ic:
                    await session.delete(count)
                    deleted_counts['image_counts'] += 1
                    
                # Find and delete faces no longer referenced
                # First, get all face_ids still referenced in image_records
                ref_stmt = select(ImageRecord.face_id).distinct()
                ref_result = await session.execute(ref_stmt)
                referenced_face_ids = [r[0] for r in ref_result]
                
                # Then find faces not in that list and with last_seen before cutoff
                face_stmt = select(Face).where(
                    Face.face_id.not_in(referenced_face_ids) if referenced_face_ids else True,
                    Face.last_seen < cutoff_date
                )
                face_result = await session.execute(face_stmt)
                old_faces = face_result.scalars().all()
                
                for face in old_faces:
                    await session.delete(face)
                    deleted_counts['faces'] += 1
                    
                # Commit all deletions
                await session.commit()
                
                return deleted_counts
        except Exception as e:
            logger.error(f"Error deleting old records: {e}")
            return {'image_records': 0, 'image_counts': 0, 'faces': 0}

    async def export_data(self, format_type="csv"):
        """
        Export face recognition data in the specified format
        
        Parameters:
        format_type (str): The export format ("csv", "json", or "excel")
        
        Returns:
        bytes: The exported data as bytes
        """
        try:
            import pandas as pd
            import io
            
            # Get face data
            async with self.session_manager.create_session() as session:
                # Faces table
                faces_stmt = select(
                    Face.face_id,
                    Face.first_seen,
                    Face.last_seen
                )
                faces_result = await session.execute(faces_stmt)
                faces_data = [
                    {
                        "face_id": row.face_id,
                        "first_seen": row.first_seen,
                        "last_seen": row.last_seen
                    }
                    for row in faces_result
                ]
                faces_df = pd.DataFrame(faces_data)
                
                # Image records
                records_stmt = select(
                    ImageRecord.record_id,
                    ImageRecord.image_path,
                    ImageRecord.face_id,
                    ImageRecord.detection_time
                )
                records_result = await session.execute(records_stmt)
                records_data = [
                    {
                        "record_id": row.record_id,
                        "image_path": row.image_path,
                        "face_id": row.face_id,
                        "detection_time": row.detection_time
                    }
                    for row in records_result
                ]
                records_df = pd.DataFrame(records_data)
                
                # Create a BytesIO object to hold the output
                output = io.BytesIO()
                
                # Export in the requested format
                if format_type.lower() == "csv":
                    # Export as CSV with multiple tables
                    faces_csv = faces_df.to_csv(index=False)
                    records_csv = records_df.to_csv(index=False)
                    
                    output.write(b"# Faces Table\n")
                    output.write(faces_csv.encode('utf-8'))
                    output.write(b"\n\n# Image Records Table\n")
                    output.write(records_csv.encode('utf-8'))
                    
                    output.seek(0)
                    return output.getvalue()
                    
                elif format_type.lower() == "json":
                    # Export as JSON
                    export_dict = {
                        "faces": faces_df.to_dict(orient="records"),
                        "image_records": records_df.to_dict(orient="records")
                    }
                    
                    import json
                    json_data = json.dumps(export_dict, default=str, indent=2)
                    return json_data.encode('utf-8')
                    
                elif format_type.lower() == "excel":
                    # Export as Excel with multiple sheets
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        faces_df.to_excel(writer, sheet_name='Faces', index=False)
                        records_df.to_excel(writer, sheet_name='ImageRecords', index=False)
                    
                    output.seek(0)
                    return output.getvalue()
                    
                else:
                    raise ValueError(f"Unsupported export format: {format_type}")
                    
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return b"Error exporting data"
# Modified database.py methods
