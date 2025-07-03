
import os
import shutil
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Form
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.file_store import save_image
from app.db.base import session_manager
from app.services.face_service import FaceService
# Needed for FaceService dependency
from app.services.image_record_service import ImageRecordService
from app.services.image_count_service import ImageCountService
from app.utils.file_store import SERVER_IMAGE_STORAGE_ROOT, PUBLIC_IMAGE_URL_PREFIX, \
    TEMP_IMAGE_STORAGE_ROOT, PUBLIC_TEMP_IMAGE_URL_PREFIX

# Pydantic models for request/response bodies
from pydantic import BaseModel, Field


class FaceDetectionReview(BaseModel):
    face_index: int
    suggested_face_id: Optional[int]
    suggested_person_name: str
    is_new_face_candidate: bool
    face_location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    face_encoding: List[float]  # Stored as list for JSON serialization


class UploadPreviewResponse(BaseModel):
    message: str
    original_filename: str
    num_files_uploaded: int
    # Each dict contains preview_image_url, detections etc.
    processed_files: List[Dict[str, Any]]


class ConfirmSaveRequest(BaseModel):
    # This structure should mirror the 'processed_files' data you send back from upload-preview
    # but potentially with user edits.
    original_image_path_server: str
    preview_image_path_server: str
    detection_time: str  # Keep as string (ISO format) for transport
    # Each dict should contain enough info for FaceService.save_confirmed_faces
    face_detections: List[Dict[str, Any]]
    batch_tag: Optional[str] = None
    # e.g., face_index, suggested_face_id, suggested_person_name, is_new_face_candidate, face_location, face_encoding
    # PLUS, if user associated: 'person_id': int (the ID of the person they chose)


class SaveConfirmationResponse(BaseModel):
    message: str
    saved_files_results: List[Dict[str, Any]]


logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency functions


async def get_db_session():
    async with session_manager.create_session() as session:
        yield session


def get_face_service(
    db: AsyncSession = Depends(get_db_session),
):
    # one session – reused by every helper service
    rec_svc = ImageRecordService(db)
    cnt_svc = ImageCountService(db)
    return FaceService(db, rec_svc, cnt_svc)

# @router.post("/uploads/images", status_code=202)
# async def upload_images(
#     files: List[UploadFile] = File(...),
#     bg: BackgroundTasks = Depends(),
#     pipeline: ImagePipeline = Depends(pipeline_dep)
# ):
#     paths = [save_image(f, "raw") for f in files]
#     bg.add_task(pipeline.process_batch, paths)
#     return {"accepted": len(paths)}


# Adjust response model as needed
@router.post("/upload-and-process", response_model=List[Dict])
async def upload_and_process_image(
    file: UploadFile = File(...),
    face_service: FaceService = Depends(get_face_service)
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    # Create a unique filename to avoid conflicts (or use original filename with path)
    # You might want to save it under a timestamped folder or with a UUID
    file_extension = os.path.splitext(file.filename)[1]
    # Example: Save with a timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"image_{timestamp}{file_extension}"

    # Define the full path where the image will be saved
    save_path = os.path.join(SERVER_IMAGE_STORAGE_ROOT, unique_filename)

    # Ensure the directory exists
    os.makedirs(SERVER_IMAGE_STORAGE_ROOT, exist_ok=True)

    try:
        # Save the uploaded file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image and store face data
        results = await face_service.process_and_store_image(save_path)

        if not results:
            # You might want to delete the file if no faces were found or it was unprocessable
            os.remove(save_path)
            return {"message": "Image uploaded, but no faces detected.", "file_path": save_path, "results": []}

        return {"message": "Image uploaded and processed successfully", "file_path": save_path, "results": results}

    except Exception as e:
        logger.error(f"Error during file upload and processing: {e}")
        # Clean up the file if processing failed
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to process image: {e}")


# Adjust response model as needed
@router.post("/upload-and-process-multiple", response_model=List[Dict])
async def upload_and_process_multiple_images(
    # Changed from 'file' to 'files' and type to List[UploadFile]
    files: List[UploadFile] = File(...),
    face_service: FaceService = Depends(get_face_service)
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    all_results = []
    # Loop through each uploaded file
    for file in files:
        file_processing_result = {
            "original_filename": file.filename,
            "status": "processing",
            "message": "",
            "results": []
        }

        if not file.filename:
            file_processing_result["status"] = "failed"
            file_processing_result["message"] = "Uploaded file has no filename."
            all_results.append(file_processing_result)
            continue  # Move to the next file

        file_extension = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # Include original name for better traceability
        unique_filename = f"uploaded_image_{timestamp}_{file.filename}"
        save_path = os.path.join(SERVER_IMAGE_STORAGE_ROOT, unique_filename)

        os.makedirs(SERVER_IMAGE_STORAGE_ROOT, exist_ok=True)

        try:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Use the existing process_and_store_image from FaceService
            results_for_current_file = await face_service.process_and_store_image(save_path)

            if not results_for_current_file:
                os.remove(save_path)  # Delete if no faces found
                file_processing_result["status"] = "no_face_detected"
                file_processing_result["message"] = "Image uploaded, but no faces detected."
            else:
                file_processing_result["status"] = "success"
                file_processing_result["message"] = "Image uploaded and processed successfully."
                file_processing_result["results"] = results_for_current_file
                # Add saved path to result
                file_processing_result["file_path"] = save_path

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)
            file_processing_result["status"] = "failed"
            file_processing_result["message"] = f"Failed to process image: {e}"

        all_results.append(file_processing_result)

    return all_results


@router.post("/upload-preview", response_model=UploadPreviewResponse)
async def upload_images_for_preview(
    files: List[UploadFile] = File(...),
    batch_tag: Optional[str] = Form(None),
    face_service: FaceService = Depends(get_face_service)
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    logger.info(f"[upload-preview] Received batch_tag={batch_tag}")

    processed_files_results = []

    # Ensure temporary storage directory exists
    os.makedirs(TEMP_IMAGE_STORAGE_ROOT, exist_ok=True)

    for file in files:
        file_result: Dict[str, Any] = {
            "original_filename": file.filename,
            "status": "processing",
            "message": "",
            "preview_image_url": None,
            "face_detections": [],
            "batch_tag": batch_tag
        }
        logger.info(f"[upload-preview] Processed file result: {file_result}")

        if not file.filename:
            file_result["status"] = "failed"
            file_result["message"] = "Uploaded file has no filename."
            processed_files_results.append(file_result)
            continue

        file_extension = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # Save original file to a temp location for further processing/moving
        unique_original_filename = f"original_{timestamp}_{file.filename}"
        temp_original_path = os.path.join(
            TEMP_IMAGE_STORAGE_ROOT, unique_original_filename)

        try:
            # Save the original uploaded file to temporary storage
            with open(temp_original_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Process the image for review (no DB save here)
            review_data = await face_service.process_image_for_review(temp_original_path)

            if review_data:
                file_result["status"] = "success"
                file_result[
                    "message"] = f"Processed {review_data['num_faces_detected']} face(s) for review."
                file_result["preview_image_url"] = review_data['preview_image_url']
                file_result["original_image_url"] = review_data['original_image_url']
                file_result["face_detections"] = review_data['face_detections']
                file_result['original_image_path_server'] = review_data['original_image_path_server']
                file_result['preview_image_path_server'] = review_data['preview_image_path_server']
                file_result['detection_time'] = review_data['detection_time']

                logger.info(
                    f"[upload-preview] Returning detection_time={file_result['detection_time']} for file={file.filename}")

            else:
                file_result["status"] = "no_face_detected"
                file_result["message"] = "Image uploaded, but no faces detected."
                # Clean up original temp file if no faces detected
                if os.path.exists(temp_original_path):
                    os.remove(temp_original_path)

        except Exception as e:
            logger.error(
                f"Error processing file {file.filename} for preview: {e}")
            if os.path.exists(temp_original_path):
                os.remove(temp_original_path)  # Clean up original temp file
            file_result["status"] = "failed"
            file_result["message"] = f"Failed to process image: {e}"

        processed_files_results.append(file_result)

    return UploadPreviewResponse(
        message="Image(s) uploaded and processed for review.",
        # Concatenate filenames if multiple
        original_filename=", ".join([f.filename for f in files]),
        num_files_uploaded=len(files),
        processed_files=processed_files_results
    )


@router.post("/confirm-save", response_model=SaveConfirmationResponse, status_code=status.HTTP_200_OK)
async def confirm_and_save_faces(
    # Expect a list of confirmations for each image
    data: List[ConfirmSaveRequest],
    face_service: FaceService = Depends(get_face_service)
):
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="No data provided for saving.")

    all_saved_results = []
    # logger.info(f"[confirm-save] Received data: {data[0]['face_detections']}")

    for confirmed_item in data:
        # Call the FaceService method that performs the actual database save
        saved_results_for_item = await face_service.save_confirmed_faces(confirmed_item.model_dump())
        all_saved_results.extend(saved_results_for_item)

    return SaveConfirmationResponse(
        message="Confirmed face detections saved to database.",
        saved_files_results=all_saved_results
    )
