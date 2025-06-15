# image_processor.py

import os
import glob
import face_recognition
import cv2
from datetime import datetime
import logging
from config import IMAGE_FOLDER

class ImageProcessor:
    def __init__(self, image_folder=None):
        self.image_folder = image_folder if image_folder else IMAGE_FOLDER

    def get_image_paths(self):
        """
        Retrieves a list of image file paths from the specified folder.
        """
        if not os.path.isdir(self.image_folder):
            logging.error(f"Image folder '{self.image_folder}' does not exist.")
            return []

        image_paths = glob.glob(os.path.join(self.image_folder, '*.*'))
        image_paths = [img for img in image_paths if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(image_paths)} images in '{self.image_folder}'.")
        return image_paths

    def load_image(self, image_path):
        """
        Loads an image file and returns it in RGB format.
        """
        try:
            image = face_recognition.load_image_file(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rgb_image
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None

    def detect_faces(self, rgb_image):
        """
        Detects face locations and encodings in an RGB image.
        Returns a tuple of (face_locations, face_encodings).
        """
        try:
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            logging.info(f"Detected {len(face_encodings)} face(s).")
            return face_locations, face_encodings
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return [], []

    def process_image(self, image_path):
        """
        Processes a single image: loads it, detects faces, and encodes them.
        Returns a dictionary with image_path, detection_time, and face_encodings.
        """
        rgb_image = self.load_image(image_path)
        if rgb_image is None:
            return None

        face_locations, face_encodings = self.detect_faces(rgb_image)
        detection_time = datetime.now()

        return {
            'image_path': image_path,
            'detection_time': detection_time,
            'face_encodings': face_encodings,
            'face_locations': face_locations,
            'rgb_image': rgb_image  # Include the image in the return data
        }

    def annotate_and_display_faces(self, rgb_image, face_locations, face_names, display=True):
        """
        Draws rectangles and annotates detected faces with names on the image.
        If 'display' is True, the image will be displayed using OpenCV's imshow.

        Args:
            rgb_image (ndarray): The RGB image where faces are detected.
            face_locations (list): A list of tuples specifying the face bounding boxes.
            face_names (list): A list of names (or IDs) corresponding to each face.
            display (bool): Whether to display the image using OpenCV's imshow.
        """
        # Convert the RGB image back to BGR for OpenCV display
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a rectangle around the face
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Annotate the face with the person's name or ID
            label = name if name else "Unknown"
            # Calculate the position for the label
            label_position = (left, top - 10) if top - 10 > 10 else (left, top + 10)
            cv2.putText(image_bgr, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show the image if display is True
        if display:
            cv2.imshow('Face Detection', image_bgr)
            cv2.waitKey(0)  # Wait for a key press to close the image
            cv2.destroyAllWindows()
