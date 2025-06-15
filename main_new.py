import logging
from image_processor import ImageProcessor
from config import FACE_DISTANCE_THRESHOLD
from app.services.face_service import FaceService
from app.services.image_record_service import ImageRecordService
from app.services.image_count_service import ImageCountService
from app.db.base import session_manager

import face_recognition
import numpy as np
import asyncio

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


async def main(use_database=True, display_faces=False):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting face processing pipeline")

    image_processor = ImageProcessor()
    known_face_ids, known_face_encodings = [], []

    try:
        image_paths = image_processor.get_image_paths()
        logger.info(f"Found {len(image_paths)} image(s) to process")
    except Exception as e:
        logger.error(f"Failed to retrieve image paths: {e}")
        return

    if use_database:
        async with session_manager.create_session() as session:
            face_service = FaceService(session)
            image_record_service = ImageRecordService(session)
            image_count_service = ImageCountService(session)

            try:
                known_face_ids, known_face_encodings = await face_service.get_all_known_faces()
                logger.info(f"Loaded {len(known_face_ids)} known face(s) from database")
            except Exception as e:
                logger.error(f"Failed to load known faces: {e}")
                return

            for image_path in image_paths:
                logger.info(f"Processing image: {image_path}")
                try:
                    result = image_processor.process_image(image_path)
                except Exception as e:
                    logger.error(f"Image processing failed for {image_path}: {e}")
                    continue

                if not result:
                    logger.warning(f"No result for image {image_path}, skipping")
                    continue

                face_encodings = result['face_encodings']
                face_locations = result['face_locations']
                rgb_image = result['rgb_image']
                current_time = result['detection_time']
                face_names = []

                for face_encoding in face_encodings:
                    face_id = None
                    try:
                        if known_face_encodings:
                            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            min_distance = min(distances)
                            if min_distance < FACE_DISTANCE_THRESHOLD:
                                index = np.argmin(distances)
                                face_id = known_face_ids[index]
                                await face_service.update_last_seen(face_id, current_time, face_encoding)
                                known_face_encodings[index] = face_encoding
                                face_names.append(f"ID: {face_id}")
                                logger.info(f"Matched existing face ID {face_id}")
                            else:
                                new_face = await face_service.insert_new_face(face_encoding, current_time)
                                face_id = new_face.face_id
                                known_face_ids.append(face_id)
                                known_face_encodings.append(face_encoding)
                                face_names.append(f"New ID: {face_id}")
                                logger.info(f"Added new face ID {face_id}")
                        else:
                            new_face = await face_service.insert_new_face(face_encoding, current_time)
                            face_id = new_face.face_id
                            known_face_ids.append(face_id)
                            known_face_encodings.append(face_encoding)
                            face_names.append(f"New ID: {face_id}")
                            logger.info(f"Added initial face ID {face_id}")

                        await image_record_service.insert_image_record(image_path, face_id, current_time)
                    except Exception as e:
                        logger.error(f"Failed handling face in {image_path}: {e}")

                if display_faces and face_encodings:
                    image_processor.annotate_and_display_faces(rgb_image, face_locations, face_names, display=True)

                try:
                    await image_count_service.insert_or_update_image_count(image_path, len(face_encodings), current_time)
                    logger.info(f"Recorded face count for {image_path}: {len(face_encodings)}")
                except Exception as e:
                    logger.error(f"Failed to record face count for {image_path}: {e}")

    else:
        for image_path in image_paths:
            logger.info(f"[NO DB] Processing image: {image_path}")
            try:
                result = image_processor.process_image(image_path)
                if not result:
                    logger.warning(f"No result for image {image_path}, skipping")
                    continue

                face_encodings = result['face_encodings']
                face_locations = result['face_locations']
                rgb_image = result['rgb_image']

                face_names = []
                for face_encoding in face_encodings:
                    temp_id = f"temp_{hash(face_encoding.tobytes())}"
                    face_names.append(f"Temp ID: {temp_id}")
                    logger.info(f"Detected face (no DB): {temp_id} in {image_path}")

                if display_faces:
                    image_processor.annotate_and_display_faces(rgb_image, face_locations, face_names, display=True)

                logger.info(f"Detected {len(face_encodings)} face(s) in {image_path}")
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")

    logger.info("Processing completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Detection and Recording Script")
    parser.add_argument('--no-db', action='store_true', help='Run without database')
    parser.add_argument('--display', action='store_true', help='Display images with annotations')

    args = parser.parse_args()
    asyncio.run(main(use_database=not args.no_db, display_faces=args.display))
