# services/face_service.py
import pytz
import numpy as np
import logging
import json
import os  # Added for path manipulation
import shutil  # Added for file copying/deleting
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload, selectinload
from typing import List, Tuple, Optional, Dict, Any
from fastapi import HTTPException, status, UploadFile
from datetime import datetime
from PIL import Image

# Assuming these imports are correctly configured in your project
from app.services.base_service import BaseService
from app.services.image_record_service import ImageRecordService
from app.db.models.attendance import Face, Person, ImageRecord
from app.schemas.common import ImageRecordOut

# Import ImageProcessor (ensure it's accessible, e.g., in the same directory or a common module)
# Adjust import path as needed
from app.functions.image_processor import ImageProcessor
from app.utils.file_store import (
    TEMP_IMAGE_STORAGE_ROOT,
    PUBLIC_TEMP_IMAGE_URL_PREFIX,
    SERVER_IMAGE_STORAGE_ROOT,
    SERVER_FACE_CROP_STORAGE_ROOT,
    PUBLIC_FACE_CROP_PREFIX,
    PUBLIC_PERSON_IMAGE_PREFIX,
)

from app.core.config import settings

# Get timezone from settings
tz = pytz.timezone(settings.TIMEZONE)

logger = logging.getLogger(__name__)


class FaceService(BaseService):
    def __init__(self, db: AsyncSession, image_record_service: ImageRecordService):
        super().__init__(db, Face)
        self.image_processor = ImageProcessor()  # Initialize ImageProcessor
        self.image_record_service = image_record_service  # Store the injected service

    async def get_all_known_faces(self) -> Tuple[List[int], List[np.ndarray]]:
        try:
            result = await self.db.execute(select(Face))
            faces = result.scalars().all()
            known_face_ids = [face.face_id for face in faces]
            known_face_encodings = [
                np.frombuffer(face.face_encoding, dtype=np.float64) for face in faces
            ]
            return known_face_ids, known_face_encodings
        except Exception as e:
            logger.error(f"Error retrieving known faces: {e}")
            return [], []

    async def insert_new_face(
        self,
        encoding: np.ndarray,
        current_time: datetime,
        person_id: Optional[int] = None,
        save_crop: bool = True,
        image_path: Optional[str] = None,
        face_location: Optional[tuple] = None,  # (top, right, bottom, left)
    ):
        try:
            encoding_bytes = encoding.tobytes()
            face = Face(
                face_encoding=encoding_bytes,
                first_seen=current_time,
                last_seen=current_time,
                person_id=person_id,
            )
            self.db.add(face)
            await self.db.commit()
            await self.db.refresh(face)

            logger.info(
                f"[insert_new_face] Face inserted with ID: {face.face_id}")
            logger.info(
                f"[insert_new_face] Face location (raw): {face_location} | type={type(face_location)}")

            # Save face crop (if image and location are available)
            if save_crop and image_path and face_location:
                logger.info(
                    f"[insert_new_face] Preparing to save face crop for face_id {face.face_id}")

                image = self.image_processor.load_image(image_path)
                if image is None:
                    logger.warning(
                        f"[insert_new_face] Failed to load image for face crop: {image_path}")
                    return face

                # Log more details about face_location
                logger.debug(
                    f"[insert_new_face] face_location: {face_location} (len={len(face_location) if hasattr(face_location, '__len__') else 'N/A'})")

                # Unpack face_location safely
                try:
                    if isinstance(face_location, (list, tuple)):
                        if len(face_location) == 4:
                            top, right, bottom, left = face_location
                        elif len(face_location) == 1 and isinstance(face_location[0], (list, tuple)) and len(face_location[0]) == 4:
                            top, right, bottom, left = face_location[0]
                        else:
                            raise ValueError(
                                f"Unsupported face_location structure: {face_location}")
                    else:
                        raise TypeError(
                            f"face_location is not list/tuple: {type(face_location)}")

                    logger.debug(
                        f"[insert_new_face] Cropping face at (top={top}, right={right}, bottom={bottom}, left={left})")

                    # Convert to integers just in case
                    cropped_face = image[int(top):int(
                        bottom), int(left):int(right)]

                    os.makedirs(PUBLIC_FACE_CROP_PREFIX, exist_ok=True)
                    filename = f"face_{face.face_id}.jpg"
                    full_path = os.path.join(
                        SERVER_FACE_CROP_STORAGE_ROOT, filename)
                    Image.fromarray(cropped_face).save(full_path)

                    logger.info(
                        f"[insert_new_face] Saved face crop at: {full_path}")

                except Exception as crop_err:
                    logger.error(
                        f"[insert_new_face] Failed during cropping/saving: {crop_err}")
                    raise

            return face

        except Exception as e:
            await self.db.rollback()
            logger.error(f"[insert_new_face] Error inserting new face: {e}")
            raise

    async def add_portrait_face(
        self,
        person_id: int,
        image_path: Optional[str] = None,
        save_crop: bool = False,
    ) -> tuple[Face, str]:
        """
        • Saves the portrait to permanent storage
        • Extracts one face encoding
        • Inserts Face with NULL first_seen / last_seen
        """
        public_url = None
        # 0. Load image
        image = self.image_processor.load_image(image_path)
        if image is None:
            raise HTTPException(400, "Failed to load image")

        # 1. Extract encoding
        res = self.image_processor.process_image(image_path)
        if not res or not res["face_encodings"]:
            raise HTTPException(400, "No face detected in portrait")

        encoding_np = res["face_encodings"][0]
        location = res["face_locations"][0]
        logger.info(f"Person ID: {person_id}")
        existing_face = await self.get_face_by_person_id(person_id)
        logger.info(f"Existing face: {existing_face}")

        if existing_face:
            # Overwrite encoding
            existing_face.face_encoding = encoding_np.tobytes()
            face = existing_face
            logger.info(f"Overwriting existing face for person_id {person_id}")
        else:
            # Insert new
            face = Face(
                face_encoding=encoding_np.tobytes(),
                person_id=person_id,
                first_seen=None,
                last_seen=None,
            )
            self.db.add(face)

        await self.db.commit()
        await self.db.refresh(face)

        # Crop and save face
        if save_crop:
            top, right, bottom, left = location
            cropped_face = image[top:bottom, left:right]
            os.makedirs(PUBLIC_FACE_CROP_PREFIX, exist_ok=True)
            filename = f"face_{face.face_id}.jpg"
            full_path = os.path.join(SERVER_FACE_CROP_STORAGE_ROOT, filename)
            Image.fromarray(cropped_face).save(full_path)

            public_url = os.path.join(PUBLIC_FACE_CROP_PREFIX, filename)

        return face, public_url

    async def update_last_seen(
        self,
        face_id: int,
        current_time: datetime,
        encoding: Optional[np.ndarray] = None,
    ):
        try:
            face = await self.get_by_id(face_id)
            if not face:
                logger.warning(
                    f"Face with ID {face_id} not found for updating last_seen."
                )
                return None
            face.last_seen = current_time
            if encoding is not None:
                face.face_encoding = encoding.tobytes()
            await self.db.commit()
            await self.db.refresh(face)
            return face
        except Exception as e:
            await self.db.rollback()
            logger.error(
                f"Error updating last_seen for face ID {face_id}: {e}")
            raise

    async def list_joined(self, known: str | None = None, person_id: int | None = None):
        stmt = select(Face, Person.name, Person.image_path).join(
            Person, Face.person_id == Person.person_id, isouter=True
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
                person_name=person_name,
                thumbnail_url=_get_face_thumbnail_url(f.face_id),
                person_image_url=_get_person_image_url(person_image_path),
            )
            for f, person_name, person_image_path in res.all()
        ]

    async def get_detail(self, face_id: int):
        stmt = (
            select(Face).options(joinedload(Face.person)).where(
                Face.face_id == face_id)
        )
        res = await self.db.execute(stmt)
        face = res.scalar_one_or_none()
        if not face:
            return None
        return {
            "face_id": face.face_id,
            "first_seen": face.first_seen,
            "last_seen": face.last_seen,
            "person_id": face.person_id,
            "person_name": face.person.name if face.person else None,
        }

    async def records_for_face(self, face_id: int) -> list[ImageRecordOut]:
        stmt = (
            select(ImageRecord)
            .options(selectinload(ImageRecord.face).selectinload(Face.person))
            .where(ImageRecord.face_id == face_id)
            .order_by(ImageRecord.detection_time.desc())
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        return [
            ImageRecordOut(
                record_id=r.record_id,
                face_id=r.face_id,
                image_path=r.image_path,
                image_url=self.image_record_service._get_image_url_from_path(
                    r.image_path),
                detection_time=r.detection_time,
                face_location=json.loads(r.face_location or "[]"),
                image_width=r.image_width,
                image_height=r.image_height,
                person_name=getattr(r.face.person, "name", None),
                batch_tag=r.batch_tag,
            )
            for r in rows
        ]

    async def associate(self, face_id: int, person_id: int):
        face = await self.get_by_id(face_id)
        if face:
            face.person_id = person_id
            await self.db.commit()
            await self.db.refresh(face)
            logger.info(
                f"Face ID {face_id} associated with Person ID {person_id}.")
        else:
            logger.warning(
                f"Face with ID {face_id} not found for association.")

    async def associate_transfer_then_delete(
        self, new_face_id: int, person_id: int
    ):
        # Face that will be kept (face_id = 22)
        new_face = await self.get_by_id(new_face_id)
        if not new_face:
            logger.warning(f"Face ID {new_face_id} not found.")
            return

        # Existing face already associated with the person (face_id = 23)
        stmt = (
            select(Face)
            .where(Face.person_id == person_id, Face.face_id != new_face_id)
        )
        res = await self.db.execute(stmt)
        old_face = res.scalars().first()

        if old_face:
            logger.info(
                f"Transferring encoding from face {old_face.face_id} to {new_face.face_id}"
            )

            try:
                copy_face_crop_image(
                    old_face.face_id, new_face.face_id, delete_original=True)
            except FileNotFoundError:
                logger.warning(
                    f"No crop image to copy from face {old_face.face_id}")
            except Exception as e:
                logger.error(f"Unexpected error during crop copy: {e}")

            if old_face.face_encoding:
                new_face.face_encoding = old_face.face_encoding

            # Assign person_id to the new face
            new_face.person_id = person_id

            # Delete the old face
            await self.db.delete(old_face)

            logger.info(f"Deleted old face {old_face.face_id}")
        else:
            # Just associate if there's no existing face for this person
            new_face.person_id = person_id

        await self.db.commit()
        await self.db.refresh(new_face)

        logger.info(
            f"Face {new_face.face_id} now associated with person {person_id}")

    async def disassociate(self, face_id: int):
        face = await self.get_by_id(face_id)
        if face:
            face.person_id = None
            await self.db.commit()
            await self.db.refresh(face)
            logger.info(f"Face ID {face_id} disassociated.")
        else:
            logger.warning(
                f"Face with ID {face_id} not found for disassociation.")

    # --- New and Modified methods for integration ---

    async def process_and_store_image(
        self, image_path: str, tolerance: float = 0.6
    ) -> List[Dict]:
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
            logger.warning(
                f"No faces detected or error processing image: {image_path}")
            return []

        detection_time = processed_data["detection_time"]
        face_encodings = processed_data["face_encodings"]
        face_locations = processed_data["face_locations"]

        results = []

        known_face_ids, known_face_encodings = await self.get_all_known_faces()

        for i, face_encoding in enumerate(face_encodings):
            match_found = False
            matched_face_id = None
            person_name = None

            if known_face_encodings:
                matches = self.image_processor.compare_faces(
                    known_face_encodings, face_encoding, tolerance
                )
                # distances = self.image_processor.get_face_distances(known_face_encodings, face_encoding)

                # Find the best match
                if True in matches:
                    first_match_index = np.where(matches)[0][0]
                    matched_face_id = known_face_ids[first_match_index]

                    # Update last_seen for the matched face
                    await self.update_last_seen(
                        matched_face_id, detection_time, face_encoding
                    )

                    # Get person name if associated
                    face_detail = await self.get_detail(matched_face_id)
                    if face_detail and face_detail["person_name"]:
                        person_name = face_detail["person_name"]
                        logger.info(
                            f"Face recognized as '{person_name}' (Face ID: {matched_face_id}) in {image_path}"
                        )
                    else:
                        logger.info(
                            f"Known face (ID: {matched_face_id}) detected in {image_path}, but not associated with a person."
                        )

                    match_found = True

            # If no match or if it's a new face, insert it
            if not match_found:
                new_face = await self.insert_new_face(face_encoding, detection_time)
                matched_face_id = new_face.face_id
                logger.info(
                    f"New anonymous face (ID: {matched_face_id}) inserted from {image_path}"
                )

            # Store ImageRecord for each detected face using ImageRecordService
            face_location_str = str(
                face_locations[i]
            )  # Convert tuple to string for storage
            image_record = await self.image_record_service.insert_image_record(
                image_path=image_path,
                detection_time=detection_time,
                face_id=matched_face_id,
                face_location=face_location_str,
            )

            results.append(
                {
                    "face_index": i,
                    "face_id": matched_face_id,
                    "person_name": person_name,
                    "is_new_face": not match_found,
                    "face_location": face_locations[i],
                    "image_record_id": image_record.record_id,
                }
            )

        return results

    # --- Core method for processing without saving to DB ---
    async def process_image_for_review(
        self, image_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a single image: loads it, detects faces, and attempts to recognize them.
        Does NOT store anything in the database at this stage.
        Generates a preview image.

        Args:
            image_path: Path to the image file (can be temporary).

        Returns:
            A dictionary with processing results and a path to an annotated preview image,
            or None if processing fails or no faces detected.
        """
        rgb_image = self.image_processor.load_image(image_path)
        if rgb_image is None:
            logger.warning(f"Could not load image for review: {image_path}")
            return None

        # shape = (height, width, channels)
        img_height, img_width = rgb_image.shape[:2]

        face_locations, face_encodings = self.image_processor.detect_faces(
            rgb_image)

        if not face_encodings:
            logger.info(f"No faces detected in image: {image_path}")
            return None

        detection_time = datetime.now(tz=tz)
        logger.info(
            f"[process_image_for_review] detection_time={detection_time} tzinfo={detection_time.tzinfo}")
        results_for_faces = []

        known_face_ids, known_face_encodings = await self.get_all_known_faces()

        face_names_for_annotation = (
            []
        )  # To collect names for annotating the preview image

        for i, face_encoding in enumerate(face_encodings):
            person_name = "Anonymous"  # Default for review
            matched_face_id = None  # No DB ID yet for new faces

            if known_face_encodings:
                matches = self.image_processor.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6
                )  # Use default tolerance or pass it
                if True in matches:
                    first_match_index = np.where(matches)[0][0]
                    matched_face_id = known_face_ids[first_match_index]

                    face_detail = await self.get_detail(matched_face_id)
                    if face_detail and face_detail["person_name"]:
                        person_name = face_detail["person_name"]
                        logger.debug(
                            f"Preview: Face matched to known person '{person_name}' (Face ID: {matched_face_id})"
                        )
                    else:
                        logger.debug(
                            f"Preview: Face matched to existing anonymous face (Face ID: {matched_face_id})"
                        )

            face_names_for_annotation.append(
                person_name
            )  # Add recognized name or "Anonymous"

            results_for_faces.append(
                {
                    "face_index": i,  # Index within this image's detections
                    "suggested_face_id": matched_face_id,  # If matched to an existing DB face
                    "suggested_person_name": person_name,
                    "is_new_face_candidate": (
                        matched_face_id is None
                    ),  # True if no match found
                    "face_location": face_locations[
                        i
                    ],  # Tuple (top, right, bottom, left)
                    # Convert numpy array to list for JSON serialization
                    "face_encoding": face_encoding.tolist(),
                    "image_width": img_width,
                    "image_height": img_height,
                }
            )

        # --- Generate and save preview image ---
        # Ensure TEMP_IMAGE_STORAGE_ROOT exists
        os.makedirs(TEMP_IMAGE_STORAGE_ROOT, exist_ok=True)

        original_filename = os.path.basename(image_path)
        logger.info(f"Original filename: {original_filename}")
        original_base_filename = original_filename.split("_")[-1]
        local_timezone = datetime.now(tz)
        # Append a unique suffix and then original name to avoid conflicts, ensure it's in temp folder
        preview_filename = f"preview_{local_timezone.strftime('%Y%m%d%H%M%S%f')}_{original_base_filename}"
        preview_image_path_server = os.path.join(
            TEMP_IMAGE_STORAGE_ROOT, preview_filename
        )
        preview_image_url = os.path.join(
            PUBLIC_TEMP_IMAGE_URL_PREFIX, preview_filename
        )  # Public URL for client
        original_image_url = os.path.join(
            PUBLIC_TEMP_IMAGE_URL_PREFIX, original_filename
        )

        annotated_image_bgr = self.image_processor.annotate_faces(
            rgb_image=rgb_image,
            face_locations=face_locations,
            face_names=face_names_for_annotation,
        )

        if self.image_processor.save_image(
            annotated_image_bgr, preview_image_path_server
        ):
            logger.info(f"Preview image saved to {preview_image_path_server}")
        else:
            logger.error(f"Failed to save preview image for {image_path}")
            # Consider returning None or raising if preview is mandatory

        return {
            # Path to the temporarily stored original file
            "original_image_path_server": image_path,
            # Path to the temporary annotated preview
            "preview_image_path_server": preview_image_path_server,
            "preview_image_url": preview_image_url,  # URL for client to view
            # URL for client to view original image
            "original_image_url": original_image_url,
            "detection_time": detection_time.isoformat(),
            "face_detections": results_for_faces,  # List of dictionaries for each face
            "num_faces_detected": len(face_encodings),
        }

    # --- New method to process confirmed data and save to DB ---
    async def save_confirmed_faces(self, confirmed_data: Dict[str, Any]) -> List[Dict]:
        """
        Saves confirmed face detection and recognition data to the database.
        Deletes temporary files after successful processing.

        Args:
            confirmed_data: A dictionary containing data from the client,
                            including original_image_path_server,
                            preview_image_path_server, detection_time,
                            and a list of face_detections (each potentially with
                            updated person_id if user associated).
        Returns:
            A list of results, each indicating success/failure for a face's save.
        """
        original_image_path_server = confirmed_data.get(
            "original_image_path_server")
        preview_image_path_server = confirmed_data.get(
            "preview_image_path_server")
        # detection_time = datetime.fromisoformat(
        #     confirmed_data["detection_time"]
        # )  # Ensure datetime object

        # --- Begin: Make detection_time timezone-aware ---
        raw_dt = confirmed_data["detection_time"]
        parsed_dt = datetime.fromisoformat(raw_dt)
        if parsed_dt.tzinfo is None:
            detection_time = tz.localize(parsed_dt)
        else:
            detection_time = parsed_dt.astimezone(tz)
        # --- End ---

        face_detections_to_save = confirmed_data.get("face_detections", [])

        saved_results = []

        # Move original image from temp to permanent storage
        permanent_image_filename = os.path.basename(original_image_path_server)
        permanent_image_path = os.path.join(
            SERVER_IMAGE_STORAGE_ROOT, permanent_image_filename
        )

        os.makedirs(
            SERVER_IMAGE_STORAGE_ROOT, exist_ok=True
        )  # Ensure permanent storage dir exists

        try:
            shutil.move(original_image_path_server, permanent_image_path)
            logger.info(
                f"Moved original image from temp to permanent: {permanent_image_path}"
            )
        except FileNotFoundError:
            logger.error(
                f"Original temp file not found: {original_image_path_server}")
            # If original not found, can't proceed. Consider raising or returning error
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Original image file not found on server.",
            )
        except Exception as e:
            logger.error(f"Error moving original image: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to move original image: {e}",
            )

        for face_data in face_detections_to_save:
            face_encoding_list = face_data["face_encoding"]
            face_encoding_np = np.array(
                face_encoding_list, dtype=np.float64
            )  # Convert back to numpy array
            # Ensure string for DB
            face_location_str = str(face_data["face_location"])

            # User might have manually associated a person_id
            person_id_from_client = face_data.get("person_id")

            # Check if this face already exists in DB based on suggested_face_id
            # Note: For new faces, suggested_face_id will be None.
            matched_face_id_from_preview = face_data.get("suggested_face_id")

            final_face_id = None
            final_person_id = None

            try:
                if matched_face_id_from_preview is not None:
                    # It's an existing face (from preview recognition)
                    # Check if client provided a new association
                    if person_id_from_client is not None:
                        # User wants to associate an existing face to a specific person
                        await self.associate(
                            matched_face_id_from_preview, person_id_from_client
                        )
                        final_person_id = person_id_from_client
                    else:
                        # Update last seen for the existing face
                        await self.update_last_seen(
                            matched_face_id_from_preview,
                            detection_time,
                            face_encoding_np,
                        )
                        # Get existing person_id from DB
                        face_obj = await self.get_by_id(matched_face_id_from_preview)
                        if face_obj:
                            final_person_id = face_obj.person_id
                    final_face_id = matched_face_id_from_preview
                else:
                    # It's a new face candidate or an unassociated existing face
                    # Insert as a new face, potentially with a person_id from client
                    new_face = await self.insert_new_face(
                        face_encoding_np,
                        detection_time,
                        person_id=person_id_from_client,
                        image_path=permanent_image_path,
                        face_location=face_data["face_location"],
                    )

                    logger.info(f"New face detected: {new_face}")
                    final_face_id = new_face.face_id
                    final_person_id = person_id_from_client

                # Create ImageRecord using ImageRecordService
                image_record = await self.image_record_service.insert_image_record(
                    image_path=permanent_image_path,  # Use permanent path
                    detection_time=detection_time,
                    face_id=final_face_id,
                    face_location=face_location_str,
                    image_width=face_data.get("image_width"),
                    image_height=face_data.get("image_height"),
                    batch_tag=face_data.get("batch_tag")
                )

                saved_results.append(
                    {
                        "face_index": face_data.get("face_index"),
                        "status": "success",
                        "face_id": final_face_id,
                        "person_id": final_person_id,
                        "image_record_id": image_record.record_id,
                        "message": "Face data saved successfully.",
                    }
                )

            except Exception as e:
                logger.error(
                    f"Error saving confirmed face data for {original_image_path_server}: {e}"
                )
                saved_results.append(
                    {
                        "face_index": face_data.get("face_index"),
                        "status": "failed",
                        "message": f"Failed to save face data: {e}",
                    }
                )

        # --- Clean up temporary files after processing all faces for the image ---
        try:
            if os.path.exists(preview_image_path_server):
                os.remove(preview_image_path_server)
                logger.info(
                    f"Deleted temporary preview image: {preview_image_path_server}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to delete temporary preview image {preview_image_path_server}: {e}"
            )

        return saved_results

    async def process_and_store_multiple_images(
        self, folder_path: Optional[str] = None, tolerance: float = 0.6
    ) -> List[List[Dict]]:
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
        if not res or not res["face_encodings"]:
            raise ValueError("No face found in supplied image")
        return np.asarray(res["face_encodings"][0])

    async def anon_face_encodings(self):
        """Return (ids, encs) for faces with person_id is NULL."""
        stmt = select(Face).where(Face.person_id.is_(None))
        res = await self.db.execute(stmt)
        faces = res.scalars().all()
        ids = [f.face_id for f in faces]
        encs = [np.frombuffer(f.face_encoding, dtype=np.float64)
                for f in faces]
        return ids, encs

    def find_matches(self, target, enc_list, threshold=0.45):
        # This method is good as is, but consider if it should also use
        # self.image_processor.compare_faces or self.image_processor.get_face_distances
        # for consistency if you want to centralize matching logic there.
        # For now, keeping it as is since it's already implemented.
        if not enc_list:
            return []
        dists = np.linalg.norm(np.stack(enc_list) - target, axis=1)
        return np.where(dists < threshold)[0]

    async def get_face_by_person_id(self, person_id: int) -> Optional[Face]:
        try:
            stmt = select(Face).where(Face.person_id == person_id)
            res = await self.db.execute(stmt)
            return res.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching face by person ID {person_id}: {e}")
            raise


# Helper
def _get_face_thumbnail_url(face_id: int) -> Optional[str]:
    filename = f"face_{face_id}.jpg"
    full_path = os.path.join(SERVER_FACE_CROP_STORAGE_ROOT, filename)
    if os.path.exists(full_path):
        return os.path.join(PUBLIC_FACE_CROP_PREFIX, filename)
    return None


def _get_person_image_url(image_path: Optional[str]) -> Optional[str]:
    """Returns the full public URL to a person's uploaded portrait."""
    if not image_path:
        return None
    return os.path.join(PUBLIC_PERSON_IMAGE_PREFIX, image_path)


def copy_face_crop_image(old_face_id: int, new_face_id: int, storage_root=SERVER_FACE_CROP_STORAGE_ROOT, delete_original: bool = False) -> None:
    """
    Copy the cropped face image from old_face_id to new_face_id.
    Supports .jpg and .png. Optionally deletes the original after copy.

    Args:
        old_face_id (int): ID of the face to copy from.
        new_face_id (int): ID of the face to copy to.
        storage_root (str): Base directory where face crops are stored.
        delete_original (bool): If True, delete the original after copying.

    Raises:
        FileNotFoundError: If the source crop image doesn't exist.
        Exception: For other unexpected file-related errors.
    """
    for ext in [".jpg", ".png"]:
        old_crop_rel_path = f"face_{old_face_id}{ext}"
        old_crop_abs_path = os.path.join(storage_root, old_crop_rel_path)
        logger.info(
            f"Copying crop image from {old_crop_abs_path} to {new_face_id}")
        if os.path.exists(old_crop_abs_path):
            new_crop_rel_path = f"face_{new_face_id}{ext}"
            new_crop_abs_path = os.path.join(storage_root, new_crop_rel_path)
            logger.info(
                f"Copying crop image from {old_crop_abs_path} to {new_crop_abs_path}")
            try:
                shutil.copy2(old_crop_abs_path, new_crop_abs_path)
                logger.info(
                    f"Copied crop image from face {old_face_id} to {new_face_id} as {ext}")

                if delete_original:
                    os.remove(old_crop_abs_path)
                    logger.info(
                        f"Deleted original crop image for face {old_face_id}")
                return  # Done, exit the function after first match

            except Exception as e:
                logger.error(
                    f"Failed to copy face crop from {old_face_id} to {new_face_id}: {e}")
                raise

    # If we reach here, no supported file was found
    raise FileNotFoundError(
        f"No crop image (.jpg or .png) found for face {old_face_id}")
