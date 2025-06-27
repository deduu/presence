
# app/utils/file_store.py
from pathlib import Path
import uuid
import shutil  # For efficient file copying from UploadFile
from fastapi import UploadFile
import aiofiles
import os

# Root directory for all application data (e.g., within your project)
APP_DATA_ROOT = os.path.join(os.getcwd(), "app_data")

# Permanent storage for processed images
SERVER_IMAGE_STORAGE_ROOT = os.path.join(APP_DATA_ROOT, "permanent_images")

# Temporary storage for images awaiting review
TEMP_IMAGE_STORAGE_ROOT = os.path.join(APP_DATA_ROOT, "temp_images")

SERVER_FACE_CROP_STORAGE_ROOT = os.path.join(APP_DATA_ROOT, "face_crops")
SERVER_PERSON_IMAGE_ROOT = os.path.join(APP_DATA_ROOT, "public_people_images")


# Public URL prefixes for serving static files
# Make sure these match how you mount StaticFiles in main.py
PUBLIC_IMAGE_URL_PREFIX = "/public_images"  # For permanent images
PUBLIC_TEMP_IMAGE_URL_PREFIX = "/public_temp_images"  # For temporary preview images

# Public URL prefixes for serving cropped faces and person images
PUBLIC_FACE_CROP_PREFIX = "/public/faces/crops"
PUBLIC_PERSON_IMAGE_PREFIX = "/public/people/images"


# Ensure directories exist on startup or during first use
os.makedirs(SERVER_IMAGE_STORAGE_ROOT, exist_ok=True)
os.makedirs(TEMP_IMAGE_STORAGE_ROOT, exist_ok=True)
os.makedirs(SERVER_FACE_CROP_STORAGE_ROOT, exist_ok=True)
os.makedirs(SERVER_PERSON_IMAGE_ROOT, exist_ok=True)


def save_image(file: "UploadFile", subdir: str) -> str:
    """
    Saves an uploaded file to a specified subdirectory and returns its public URL path.

    Args:
        file: The UploadFile object from FastAPI (or similar file-like object).
              It must have .filename and .file (a file-like object).
        subdir: The subdirectory within the SERVER_IMAGE_STORAGE_ROOT to save the file.

    Returns:
        The URL path that the frontend can use to access the saved image.
    """
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")

    # 1. Sanitize and make filename unique
    original_extension = Path(file.filename).suffix
    # Use UUID for uniqueness
    unique_filename = f"{uuid.uuid4()}{original_extension}"

    # Construct the full filesystem path where the file will be saved
    dest_dir = SERVER_IMAGE_STORAGE_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    file_system_path = dest_dir / unique_filename

    # 2. Read and write the file in chunks to prevent memory issues
    try:
        with file_system_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Alternatively, if you prefer explicit chunking:
        # with file_system_path.open("wb") as buffer:
        #     while True:
        #         chunk = file.file.read(8192) # Read in 8KB chunks
        #         if not chunk:
        #             break
        #         buffer.write(chunk)
    except Exception as e:
        # Handle potential file writing errors
        print(f"Error saving file: {e}")
        raise  # Re-raise the exception or handle appropriately

    # 3. Return the URL path for database storage and frontend use
    # This path is relative to the PUBLIC_IMAGE_URL_PREFIX
    url_path = f"{PUBLIC_IMAGE_URL_PREFIX}/{subdir}/{unique_filename}"

    return url_path


async def save_person_image_async(file: UploadFile, person_id: int) -> str:
    from pathlib import Path
    import os
    import uuid

    subdir = f"people/{person_id}"
    filename = f"{uuid.uuid4().hex}{Path(file.filename).suffix}"

    directory = Path("app_data/public_people_images") / subdir
    os.makedirs(directory, exist_ok=True)

    full_path = directory / filename

    try:
        async with aiofiles.open(full_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise Exception(f"Failed saving image file: {str(e)}")

    return f"/public/people/images/{subdir}/{filename}", full_path


# # app/utils/file_store.py
# from pathlib import Path, PurePath
# BASE = Path("data/images")    # create this directory

# def save_image(file, subdir: str) -> str:
#     dest_dir = BASE / subdir
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     filename = dest_dir / file.filename
#     with filename.open("wb") as f:
#         f.write(file.file.read())
#     return str(filename)      # absolute path stored in DB
