# # services/face_service.py

# import numpy as np
# import logging

# from sqlalchemy.ext.asyncio import AsyncSession

# from sqlalchemy import select
# from typing import List, Tuple, Optional

# from datetime import datetime
# from sqlalchemy import select, func
# from sqlalchemy.orm import joinedload

# from app.services.base_service import BaseService
# from app.db.models.attendance import Face, Person, ImageRecord

# logger = logging.getLogger(__name__)

# class FaceService(BaseService):
#     def __init__(self, db: AsyncSession):
#         super().__init__(db, Face)

    
#     async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
#         try:
#             result = await self.db.execute(select(Face))
#             faces = result.scalars().all()
#             known_face_ids = [face.face_id for face in faces]
#             known_face_encodings = [np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces]
#             return known_face_ids, known_face_encodings
#         except Exception as e:
#             logger.error(f"Error retrieving known faces: {e}")
#             return [], []
    
#     async def insert_new_face(self, encoding: np.ndarray, current_time: datetime):
#         try:
#             encoding_bytes = encoding.tobytes()
#             face = Face(face_encoding=encoding_bytes, first_seen=current_time, last_seen=current_time)
#             self.db.add(face)
#             await self.db.commit()
#             await self.db.refresh(face)
#             return face
#         except Exception as e:
#             await self.db.rollback()
#             logger.error(f"Error inserting new face: {e}")
#             raise

#     async def update_last_seen(self, face_id: int, current_time: datetime, encoding: Optional[np.ndarray] = None):
#         try:
#             face = await self.get_by_id(face_id)
#             face.last_seen = current_time
#             if encoding is not None:
#                 face.face_encoding = encoding.tobytes()
#             await self.db.commit()
#             await self.db.refresh(face)
#             return face
#         except Exception as e:
#             await self.db.rollback()
#             logger.error(f"Error updating last_seen for face ID {face_id}: {e}")
#             raise
    
#     async def list_joined(self, known: str | None = None, person_id: int | None = None):
#         stmt = (
#             select(Face, Person.name)
#             .join(Person, Face.person_id == Person.person_id, isouter=True)
#         )
#         if known == "known":
#             stmt = stmt.where(Face.person_id.is_not(None))
#         elif known == "anonymous":
#             stmt = stmt.where(Face.person_id.is_(None))
#         if person_id:
#             stmt = stmt.where(Face.person_id == person_id)

#         res = await self.db.execute(stmt)
#         return [
#             dict(
#                 face_id=f.face_id,
#                 first_seen=f.first_seen,
#                 last_seen=f.last_seen,
#                 person_id=f.person_id,
#                 person_name=name,
#             )
#             for f, name in res.all()
#         ]

#     async def get_detail(self, face_id: int):
#         stmt = select(Face).options(joinedload(Face.person)).where(Face.face_id == face_id)
#         res  = await self.db.execute(stmt)
#         face = res.scalar_one_or_none()
#         if not face:
#             return None
#         return {
#             "face_id": face.face_id,
#             "first_seen": face.first_seen,
#             "last_seen":  face.last_seen,
#             "person_id":  face.person_id,
#             "person_name": face.person.name if face.person else None,
#         }

#     async def records_for_face(self, face_id: int):
#         stmt = (
#             select(ImageRecord)
#             .where(ImageRecord.face_id == face_id)
#             .order_by(ImageRecord.detection_time.desc())
#         )
#         res = await self.db.execute(stmt)
#         return res.scalars().all()

#     async def associate(self, face_id: int, person_id: int):
#         face = await self.get_by_id(face_id)
#         face.person_id = person_id
#         await self.db.commit()

#     async def disassociate(self, face_id: int):
#         face = await self.get_by_id(face_id)
#         face.person_id = None
#         await self.db.commit()

#         # --- below your existing methods ---

#     async def extract_encoding_from_path(self, img_path: str):
#         from image_processor import ImageProcessor
#         ip = ImageProcessor()
#         res = ip.process_image(img_path)
#         if not res or not res['face_encodings']:
#             raise ValueError("No face found in supplied image")
#         return np.asarray(res['face_encodings'][0])

#     async def anon_face_encodings(self):
#         """Return (ids, encs) for faces with person_id is NULL."""
#         stmt = select(Face).where(Face.person_id.is_(None))
#         res  = await self.db.execute(stmt)
#         faces = res.scalars().all()
#         ids  = [f.face_id for f in faces]
#         encs = [np.frombuffer(f.face_encoding, dtype=np.float64) for f in faces]
#         return ids, encs

#     def find_matches(self, target, enc_list, threshold=0.45):
#         import numpy as np
#         if not enc_list: return []
#         dists = np.linalg.norm(np.stack(enc_list) - target, axis=1)
#         return np.where(dists < threshold)[0]
# services/face_service.py

import numpy as np
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload
from typing import List, Tuple, Optional, Dict

from datetime import datetime

# Assuming these imports are correctly configured in your project
from app.services.base_service import BaseService
from app.services.image_record_service import ImageRecordService
from app.db.models.attendance import Face, Person, ImageRecord

# Import ImageProcessor (ensure it's accessible, e.g., in the same directory or a common module)
from app.functions.image_processor import ImageProcessor # Adjust import path as needed

logger = logging.getLogger(__name__)

class FaceService(BaseService):
    def __init__(self, db: AsyncSession, image_record_service: ImageRecordService):
        super().__init__(db, Face)
        self.image_processor = ImageProcessor() # Initialize ImageProcessor
        self.image_record_service = image_record_service # Store the injected service
        
    async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
        try:
            result = await self.db.execute(select(Face))
            faces = result.scalars().all()
            known_face_ids = [face.face_id for face in faces]
            known_face_encodings = [np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces]
            return known_face_ids, known_face_encodings
        except Exception as e:
            logger.error(f"Error retrieving known faces: {e}")
            return [], []
    
    async def insert_new_face(self, encoding: np.ndarray, current_time: datetime, person_id: Optional[int] = None):
        try:
            encoding_bytes = encoding.tobytes()
            face = Face(face_encoding=encoding_bytes, first_seen=current_time, last_seen=current_time, person_id=person_id)
            self.db.add(face)
            await self.db.commit()
            await self.db.refresh(face)
            return face
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error inserting new face: {e}")
            raise

    async def update_last_seen(self, face_id: int, current_time: datetime, encoding: Optional[np.ndarray] = None):
        try:
            face = await self.get_by_id(face_id)
            if not face:
                logger.warning(f"Face with ID {face_id} not found for updating last_seen.")
                return None
            face.last_seen = current_time
            if encoding is not None:
                face.face_encoding = encoding.tobytes()
            await self.db.commit()
            await self.db.refresh(face)
            return face
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating last_seen for face ID {face_id}: {e}")
            raise
    
    async def list_joined(self, known: str | None = None, person_id: int | None = None):
        stmt = (
            select(Face, Person.name)
            .join(Person, Face.person_id == Person.person_id, isouter=True)
        )
        if known == "known":
            stmt = stmt.where(Face.person_id.is_not(None))
        elif known == "anonymous":
            stmt = stmt.where(Face.person_id.is_(None))
        if person_id:
            stmt = stmt.where(Face.person_id == person_id)

        res = await self.db.execute(stmt)
        return [
            dict(
                face_id=f.face_id,
                first_seen=f.first_seen,
                last_seen=f.last_seen,
                person_id=f.person_id,
                person_name=name,
            )
            for f, name in res.all()
        ]

    async def get_detail(self, face_id: int):
        stmt = select(Face).options(joinedload(Face.person)).where(Face.face_id == face_id)
        res  = await self.db.execute(stmt)
        face = res.scalar_one_or_none()
        if not face:
            return None
        return {
            "face_id": face.face_id,
            "first_seen": face.first_seen,
            "last_seen":  face.last_seen,
            "person_id":  face.person_id,
            "person_name": face.person.name if face.person else None,
        }

    async def records_for_face(self, face_id: int):
        stmt = (
            select(ImageRecord)
            .where(ImageRecord.face_id == face_id)
            .order_by(ImageRecord.detection_time.desc())
        )
        res = await self.db.execute(stmt)
        return res.scalars().all()

    async def associate(self, face_id: int, person_id: int):
        face = await self.get_by_id(face_id)
        if face:
            face.person_id = person_id
            await self.db.commit()
            await self.db.refresh(face)
            logger.info(f"Face ID {face_id} associated with Person ID {person_id}.")
        else:
            logger.warning(f"Face with ID {face_id} not found for association.")

    async def disassociate(self, face_id: int):
        face = await self.get_by_id(face_id)
        if face:
            face.person_id = None
            await self.db.commit()
            await self.db.refresh(face)
            logger.info(f"Face ID {face_id} disassociated.")
        else:
            logger.warning(f"Face with ID {face_id} not found for disassociation.")

    # --- New and Modified methods for integration ---

    async def process_and_store_image(self, image_path: str, tolerance: float = 0.6) -> List[Dict]:
        """
        Processes an image, detects faces, attempts to recognize them,
        and stores relevant information in the database.
        
        Args:
            image_path: Path to the image file.
            tolerance: Distance tolerance for face matching.
            
        Returns:
            A list of dictionaries, each containing recognition results for a detected face.
        """
        processed_data = self.image_processor.process_image(image_path)
        if not processed_data:
            logger.warning(f"No faces detected or error processing image: {image_path}")
            return []

        detection_time = processed_data['detection_time']
        face_encodings = processed_data['face_encodings']
        face_locations = processed_data['face_locations']
        
        results = []
        
        known_face_ids, known_face_encodings = await self.get_all_known_faces()

        for i, face_encoding in enumerate(face_encodings):
            match_found = False
            matched_face_id = None
            person_name = None

            if known_face_encodings:
                matches = self.image_processor.compare_faces(known_face_encodings, face_encoding, tolerance)
                # distances = self.image_processor.get_face_distances(known_face_encodings, face_encoding)
                
                # Find the best match
                if True in matches:
                    first_match_index = np.where(matches)[0][0]
                    matched_face_id = known_face_ids[first_match_index]
                    
                    # Update last_seen for the matched face
                    await self.update_last_seen(matched_face_id, detection_time, face_encoding)
                    
                    # Get person name if associated
                    face_detail = await self.get_detail(matched_face_id)
                    if face_detail and face_detail['person_name']:
                        person_name = face_detail['person_name']
                        logger.info(f"Face recognized as '{person_name}' (Face ID: {matched_face_id}) in {image_path}")
                    else:
                        logger.info(f"Known face (ID: {matched_face_id}) detected in {image_path}, but not associated with a person.")
                    
                    match_found = True
            
            # If no match or if it's a new face, insert it
            if not match_found:
                new_face = await self.insert_new_face(face_encoding, detection_time)
                matched_face_id = new_face.face_id
                logger.info(f"New anonymous face (ID: {matched_face_id}) inserted from {image_path}")
            
            # Store ImageRecord for each detected face using ImageRecordService
            face_location_str = str(face_locations[i]) # Convert tuple to string for storage
            image_record = await self.image_record_service.insert_image_record(
                image_path=image_path,
                detection_time=detection_time,
                face_id=matched_face_id,
                face_location=face_location_str
            )
            
            results.append({
                'face_index': i,
                'face_id': matched_face_id,
                'person_name': person_name,
                'is_new_face': not match_found,
                'face_location': face_locations[i],
                'image_record_id': image_record.record_id
            })
            
        return results

    async def process_and_store_multiple_images(self, folder_path: Optional[str] = None, tolerance: float = 0.6) -> List[List[Dict]]:
        """
        Processes all images in a given folder (or the default), detects, recognizes,
        and stores face information in the database.
        
        Args:
            folder_path: Path to the folder containing images. If None, uses ImageProcessor's default.
            tolerance: Distance tolerance for face matching.
            
        Returns:
            A list of lists, where each inner list contains results for faces in one image.
        """
        if folder_path:
            self.image_processor.set_image_folder(folder_path)
        
        image_paths = self.image_processor.get_image_paths()
        
        all_image_results = []
        for image_path in image_paths:
            logger.info(f"Processing image: {image_path}")
            image_results = await self.process_and_store_image(image_path, tolerance)
            all_image_results.append(image_results)
        
        logger.info(f"Finished processing {len(image_paths)} images.")
        return all_image_results

    async def extract_encoding_from_path(self, img_path: str):
        # This method can now directly use the internal image_processor
        res = self.image_processor.process_image(img_path)
        if not res or not res['face_encodings']:
            raise ValueError("No face found in supplied image")
        return np.asarray(res['face_encodings'][0])

    async def anon_face_encodings(self):
        """Return (ids, encs) for faces with person_id is NULL."""
        stmt = select(Face).where(Face.person_id.is_(None))
        res  = await self.db.execute(stmt)
        faces = res.scalars().all()
        ids  = [f.face_id for f in faces]
        encs = [np.frombuffer(f.face_encoding, dtype=np.float64) for f in faces]
        return ids, encs

    def find_matches(self, target, enc_list, threshold=0.45):
        # This method is good as is, but consider if it should also use
        # self.image_processor.compare_faces or self.image_processor.get_face_distances
        # for consistency if you want to centralize matching logic there.
        # For now, keeping it as is since it's already implemented.
        if not enc_list: return []
        dists = np.linalg.norm(np.stack(enc_list) - target, axis=1)
        return np.where(dists < threshold)[0]