
# image_processor.py

import os
import glob
import face_recognition
import cv2
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional, Union


class ImageProcessor:
    """
    A modular image processing class for face detection and recognition tasks.
    Designed to be easily reusable across different scripts without external dependencies.
    """
    
    def __init__(self, 
                 image_folder: Optional[str] = None, 
                 supported_extensions: Optional[Tuple[str, ...]] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the ImageProcessor.
        
        Args:
            image_folder: Path to the folder containing images. Defaults to current directory.
            supported_extensions: Tuple of supported file extensions. 
            log_level: Logging level for the processor.
        """
        self.image_folder = image_folder or os.getcwd()
        self.supported_extensions = supported_extensions or ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        
        # Setup logging
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_image_paths(self, folder_path: Optional[str] = None) -> List[str]:
        """
        Retrieves a list of image file paths from the specified folder.
        
        Args:
            folder_path: Optional path to search. Uses instance folder if not provided.
            
        Returns:
            List of valid image file paths.
        """
        search_folder = folder_path or self.image_folder
        
        if not os.path.isdir(search_folder):
            self.logger.error(f"Image folder '{search_folder}' does not exist.")
            return []

        # Get all files in the directory
        all_files = glob.glob(os.path.join(search_folder, '*.*'))
        
        # Filter by supported extensions (case-insensitive)
        image_paths = [
            img for img in all_files 
            if any(img.lower().endswith(ext.lower()) for ext in self.supported_extensions)
        ]
        
        self.logger.info(f"Found {len(image_paths)} images in '{search_folder}'.")
        return image_paths

    def load_image(self, image_path: str) -> Optional[object]:
        """
        Loads an image file and returns it in RGB format.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            RGB image array or None if loading fails.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file '{image_path}' does not exist.")
            return None
            
        try:
            # Load image using face_recognition (which handles various formats)
            image = face_recognition.load_image_file(image_path)
            self.logger.debug(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None

    def detect_faces(self, rgb_image: object) -> Tuple[List, List]:
        """
        Detects face locations and encodings in an RGB image.
        
        Args:
            rgb_image: RGB image array.
            
        Returns:
            Tuple of (face_locations, face_encodings).
        """
        try:
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            self.logger.info(f"Detected {len(face_encodings)} face(s).")
            return face_locations, face_encodings
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return [], []

    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Processes a single image: loads it, detects faces, and encodes them.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with processing results or None if processing fails.
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
            'rgb_image': rgb_image,
            'num_faces': len(face_encodings)
        }

    def process_multiple_images(self, image_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths. If None, processes all images in the folder.
            
        Returns:
            List of processing results for each image.
        """
        if image_paths is None:
            image_paths = self.get_image_paths()
        
        results = []
        for image_path in image_paths:
            result = self.process_image(image_path)
            if result:
                results.append(result)
        
        self.logger.info(f"Successfully processed {len(results)} out of {len(image_paths)} images.")
        return results

    def annotate_faces(self, 
                      rgb_image: object, 
                      face_locations: List, 
                      face_names: List[str],
                      box_color: Tuple[int, int, int] = (0, 255, 0),
                      text_color: Tuple[int, int, int] = (255, 255, 255),
                      font_scale: float = 0.9,
                      box_thickness: int = 2) -> object:
        """
        Draws rectangles and annotates detected faces with names on the image.
        
        Args:
            rgb_image: The RGB image where faces are detected.
            face_locations: List of tuples specifying the face bounding boxes.
            face_names: List of names corresponding to each face.
            box_color: BGR color tuple for the bounding box.
            text_color: BGR color tuple for the text.
            font_scale: Scale factor for the font.
            box_thickness: Thickness of the bounding box lines.
            
        Returns:
            Annotated image in BGR format (ready for OpenCV display).
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw rectangle around the face
            cv2.rectangle(image_bgr, (left, top), (right, bottom), box_color, box_thickness)
            
            # Prepare label
            label = str(name) if name else "Unknown"
            
            # Calculate label position (above the box if possible, otherwise below)
            label_position = (left, top - 10) if top - 10 > 10 else (left, bottom + 25)
            
            # Add text
            cv2.putText(image_bgr, label, label_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
        
        return image_bgr

    def display_image(self, 
                     image: object, 
                     window_name: str = "Image Display", 
                     wait_key: bool = True,
                     destroy_on_close: bool = True) -> None:
        """
        Display an image using OpenCV.
        
        Args:
            image: Image to display (should be in BGR format for OpenCV).
            window_name: Name of the display window.
            wait_key: Whether to wait for key press before continuing.
            destroy_on_close: Whether to destroy windows after key press.
        """
        cv2.imshow(window_name, image)
        
        if wait_key:
            cv2.waitKey(0)
        
        if destroy_on_close:
            cv2.destroyAllWindows()

    def save_image(self, image: object, output_path: str) -> bool:
        """
        Save an image to disk.
        
        Args:
            image: Image to save (BGR format).
            output_path: Path where to save the image.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            success = cv2.imwrite(output_path, image)
            if success:
                self.logger.info(f"Image saved successfully to: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save image to: {output_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error saving image to {output_path}: {e}")
            return False

    def compare_faces(self, 
                     known_encodings: List, 
                     face_encoding: object, 
                     tolerance: float = 0.6) -> List[bool]:
        """
        Compare a face encoding against known face encodings.
        
        Args:
            known_encodings: List of known face encodings.
            face_encoding: Face encoding to compare.
            tolerance: Distance tolerance for face matching.
            
        Returns:
            List of boolean values indicating matches.
        """
        if not known_encodings:
            return []
        
        try:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=tolerance
            )
            return matches
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return []

    def get_face_distances(self, known_encodings: List, face_encoding: object) -> List[float]:
        """
        Calculate face distances between known encodings and a face encoding.
        
        Args:
            known_encodings: List of known face encodings.
            face_encoding: Face encoding to compare.
            
        Returns:
            List of distances (lower means more similar).
        """
        if not known_encodings:
            return []
        
        try:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            return distances.tolist()
        except Exception as e:
            self.logger.error(f"Error calculating face distances: {e}")
            return []

    def set_image_folder(self, new_folder: str) -> bool:
        """
        Change the default image folder.
        
        Args:
            new_folder: Path to the new image folder.
            
        Returns:
            True if folder exists and was set, False otherwise.
        """
        if os.path.isdir(new_folder):
            self.image_folder = new_folder
            self.logger.info(f"Image folder changed to: {new_folder}")
            return True
        else:
            self.logger.error(f"Folder '{new_folder}' does not exist.")
            return False

    def get_image_info(self, image_path: str) -> Optional[Dict]:
        """
        Get basic information about an image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with image information or None if file doesn't exist.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file '{image_path}' does not exist.")
            return None
        
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            height, width, channels = image.shape
            file_size = os.path.getsize(image_path)
            
            return {
                'path': image_path,
                'filename': os.path.basename(image_path),
                'width': width,
                'height': height,
                'channels': channels,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            self.logger.error(f"Error getting image info for {image_path}: {e}")
            return None