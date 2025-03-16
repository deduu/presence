# main.py

import logging
from image_processor import ImageProcessor
from config import FACE_DISTANCE_THRESHOLD
import face_recognition
import numpy as np
import asyncio

from app.crud.attendance import *
def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

async def main(use_database=True, display_faces=False):
    setup_logging()

    # Initialize image processor
    image_processor = ImageProcessor()

    # Initialize database handler if needed
    db_handler = None
    if use_database:
        db_handler = AsyncDatabaseHandler()
        await db_handler.connect()
        known_face_ids, known_face_encodings = await db_handler.get_all_known_faces()
    else:
        known_face_ids, known_face_encodings = [], []

    # Get list of images to process
    image_paths = image_processor.get_image_paths()

    for image_path in image_paths:
        logging.info(f"Processing image: {image_path}")
        result = image_processor.process_image(image_path)
        if not result:
            continue

        face_encodings = result['face_encodings']
        face_locations = result['face_locations']
        rgb_image = result['rgb_image']
        face_ids_in_image = []
        current_time = result['detection_time']

        face_names = []  # Initialize list to store face names or IDs

        for face_encoding in face_encodings:
            face_id = None

            if use_database and known_face_encodings:
                # Compare with known faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                min_distance = min(distances)
                if min_distance < FACE_DISTANCE_THRESHOLD:
                    # Match found
                    index = np.argmin(distances)
                    face_id = known_face_ids[index]
                    # Update last_seen and face_encoding
                    await db_handler.update_last_seen(face_id, current_time, face_encoding)
                    known_face_encodings[index] = face_encoding  # Update encoding in memory
                    logging.info(f"Matched face ID {face_id} in {image_path}.")
                    face_names.append(f"ID: {face_id}")
                else:
                    # No match found, create new face entry
                    face_id = await db_handler.insert_new_face(face_encoding, current_time)
                    if face_id:
                        known_face_ids.append(face_id)
                        known_face_encodings.append(face_encoding)
                        face_names.append(f"New ID: {face_id}")
            elif use_database and not known_face_encodings:
                # No known faces, add the first one
                face_id = await db_handler.insert_new_face(face_encoding, current_time)
                if face_id:
                    known_face_ids.append(face_id)
                    known_face_encodings.append(face_encoding)
                    face_names.append(f"New ID: {face_id}")

            if use_database and face_id:
                face_ids_in_image.append(face_id)
                # Insert into image_records
                await db_handler.insert_image_record(result['image_path'], face_id, current_time)
            else:
                # If not using database, assign a temporary unique identifier
                temp_id = f"temp_{hash(face_encoding.tobytes())}"
                face_names.append(f"Temp ID: {temp_id}")
                face_ids_in_image.append(temp_id)
                logging.info(f"Detected face (no DB): {temp_id} in {image_path}")

        # Annotate and display the faces if the flag is set
        if display_faces and face_encodings:
            image_processor.annotate_and_display_faces(rgb_image, face_locations, face_names, display=True)

        # Record image face count
        face_count = len(face_encodings)
        if use_database:
            await db_handler.insert_or_update_image_count(result['image_path'], face_count, current_time)
        else:
            logging.info(f"Image: {image_path}, Face Count: {face_count}")

        logging.info(f"Detected {face_count} face(s) in {image_path}.")

    # Close the database connection
    if use_database and db_handler:
        await db_handler.close()

    logging.info("Processing completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Detection and Recording Script")
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Run the script without connecting to the database.'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Enable face annotation and display images.'
    )

    args = parser.parse_args()
    asyncio.run(main(use_database=not args.no_db, display_faces=args.display))
