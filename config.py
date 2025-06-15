# config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'your_database_name'),
    'user': os.getenv('DB_USER', 'your_username'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Folder containing images
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER', 'myimage')

# Threshold for face recognition
FACE_DISTANCE_THRESHOLD = float(os.getenv('FACE_DISTANCE_THRESHOLD', '0.6'))

# Visualization settings
VISUALIZE = bool(int(os.getenv('VISUALIZE', '0')))  # 1 to enable, 0 to disable
