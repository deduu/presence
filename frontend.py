import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import face_recognition
import cv2
import requests  # <-- using requests to call your FastAPI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import io
import base64
from PIL import Image
import time

from image_processor import ImageProcessor
from config import FACE_DISTANCE_THRESHOLD

##############################################################################
# Set your FastAPI base URL
##############################################################################
API_BASE_URL = "http://127.0.0.1:8000/api"  # adjust if needed

##############################################################################
# Streamlit Page Config & Logging
##############################################################################
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

##############################################################################
# Session State Initialization
##############################################################################
# Initialize mode in session state if not present
if 'mode' not in st.session_state:
    st.session_state.mode = "Single Image"

if 'known_face_ids' not in st.session_state:
    st.session_state.known_face_ids = []
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = ImageProcessor()

##############################################################################
# Helper Functions to Call Your FastAPI Endpoints
##############################################################################
def test_api_connection() -> bool:
    """
    Test connectivity by calling an endpoint, e.g. get known faces.
    If it succeeds, we consider ourselves 'connected'.
    """
    try:
        url = f"{API_BASE_URL}/faces/known"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Could not connect to API: {str(e)}")
        return False

def get_all_known_faces():
    """
    Calls GET /api/faces/known to retrieve face_ids and face_encodings.
    """
    url = f"{API_BASE_URL}/faces/known"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["face_ids"], data["face_encodings"]

def insert_new_face(face_encoding: np.ndarray):
    """
    Calls POST /api/faces with a list of floats for the encoding.
    Returns the new face_id.
    """
    url = f"{API_BASE_URL}/faces"
    payload = {"encoding": face_encoding.tolist()}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["face_id"]

def update_face_encoding(face_id: int, face_encoding: np.ndarray = None):
    """
    Calls PATCH /api/faces/{face_id} to update last_seen (and optionally encoding).
    """
    url = f"{API_BASE_URL}/faces/{face_id}"
    if face_encoding is not None:
        payload = {"encoding": face_encoding.tolist()}
    else:
        payload = {"encoding": []}  # or an empty list if no new encoding
    resp = requests.patch(url, json=payload)
    resp.raise_for_status()

def insert_image_record(image_path: str, face_id: int):
    """
    Calls POST /api/images/record to log an image for a face_id.
    """
    url = f"{API_BASE_URL}/images/record"
    payload = {"image_path": image_path, "face_id": face_id}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

def insert_image_count(image_path: str, face_count: int):
    """
    Calls POST /api/images/count to insert or update face_count for an image.
    """
    url = f"{API_BASE_URL}/images/count"
    payload = {"image_path": image_path, "face_count": face_count}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

def get_image_count() -> int:
    """
    Calls GET /api/analytics/image-count to fetch total images.
    """
    url = f"{API_BASE_URL}/analytics/image-count"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("count", 0)

def get_face_appearance_data():
    """
    Calls GET /api/analytics/face-appearance to get face appearances.
    Returns a list of dicts with keys: face_id, detection_time, image_path.
    """
    url = f"{API_BASE_URL}/analytics/face-appearance"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]

def get_face_details_batch(face_ids):
    """
    Calls POST /api/analytics/face-details-batch with a list of face_ids.
    Returns a dict keyed by face_id with details about each face.
    """
    url = f"{API_BASE_URL}/analytics/face-details-batch"
    resp = requests.post(url, json=face_ids)
    resp.raise_for_status()
    data = resp.json()
    return data["details"]

def get_face_co_occurrences(face_ids):
    """
    Calls POST /api/analytics/co-occurrences to get co-occurrence matrix.
    """
    url = f"{API_BASE_URL}/analytics/co-occurrences"
    resp = requests.post(url, json=face_ids)
    resp.raise_for_status()
    data = resp.json()
    return data["co_occurrence_matrix"]

def get_face_time_distribution(face_id=None):
    """
    Calls GET /api/analytics/time-distribution?face_id=XYZ (optional).
    """
    url = f"{API_BASE_URL}/analytics/time-distribution"
    params = {}
    if face_id:
        params["face_id"] = face_id
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def delete_old_records(cutoff_date: str):
    """
    Calls DELETE /api/admin/cleanup?cutoff_date=YYYY-MM-DD
    """
    url = f"{API_BASE_URL}/admin/cleanup"
    params = {"cutoff_date": cutoff_date}
    resp = requests.delete(url, params=params)
    resp.raise_for_status()
    return resp.json()

def export_data(format_type: str) -> bytes:
    """
    Calls GET /api/admin/export?format_type=...
    Returns the raw bytes (CSV, JSON, or Excel).
    """
    url = f"{API_BASE_URL}/admin/export"
    params = {"format_type": format_type.lower()}
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()  # { "file_content": "...", "file_format": "csv" } etc.
    base64_str = data["file_content"]
    return base64.b64decode(base64_str)

##############################################################################
# Streamlit Utility Functions (Local Only)
##############################################################################
def get_image_with_faces(image_path, face_locations, face_names):
    """Draw rectangles and names on faces in the image."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(rgb_image, (left, bottom), (right, bottom + 35), (0, 255, 0), cv2.FILLED)
        cv2.putText(rgb_image, name, (left + 6, bottom + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return rgb_image

def image_to_base64(image_array):
    """Convert image array to base64 string for display."""
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

##############################################################################
# Core Image Processing Function (Uses the API for DB ops)
##############################################################################
def process_single_image(image_path, use_database=True):
    """
    Process a single image using face_recognition.  
    1) Detect face encodings/locations
    2) If using DB, match or insert new faces via the API
    3) Return result dict
    """
    result = st.session_state.image_processor.process_image(image_path)
    if not result:
        return None
    
    face_encodings = result['face_encodings']
    face_locations = result['face_locations']
    current_time = result['detection_time']
    face_names = []
    face_ids_in_image = []
    
    for face_encoding in face_encodings:
        face_id = None
        
        if use_database and st.session_state.known_face_encodings:
            # Compare with known faces
            distances = face_recognition.face_distance(
                [np.array(enc, dtype=np.float64) for enc in st.session_state.known_face_encodings],
                face_encoding
            )
            min_distance = min(distances) if len(distances) > 0 else 1.0
            if min_distance < FACE_DISTANCE_THRESHOLD:
                index = np.argmin(distances)
                face_id = st.session_state.known_face_ids[index]
                # Update last_seen (and possibly the encoding) via API
                update_face_encoding(face_id, face_encoding)
                # Keep local cache in sync
                st.session_state.known_face_encodings[index] = face_encoding.tolist()
                face_names.append(f"ID: {face_id}")
            else:
                # Insert new face via API
                face_id = insert_new_face(face_encoding)
                st.session_state.known_face_ids.append(face_id)
                st.session_state.known_face_encodings.append(face_encoding.tolist())
                face_names.append(f"New ID: {face_id}")
        
        elif use_database and not st.session_state.known_face_encodings:
            # DB enabled but no known faces yet
            face_id = insert_new_face(face_encoding)
            st.session_state.known_face_ids.append(face_id)
            st.session_state.known_face_encodings.append(face_encoding.tolist())
            face_names.append(f"New ID: {face_id}")
        
        if use_database and face_id:
            face_ids_in_image.append(face_id)
            insert_image_record(result['image_path'], face_id)
        else:
            temp_id = f"temp_{hash(face_encoding.tobytes())}"
            face_names.append(f"Temp ID: {temp_id[:8]}")
            face_ids_in_image.append(temp_id)
    
    face_count = len(face_encodings)
    if use_database:
        insert_image_count(result['image_path'], face_count)
    
    annotated_image = get_image_with_faces(image_path, face_locations, face_names)
    
    return {
        'image_path': image_path,
        'face_count': face_count,
        'face_ids': face_ids_in_image,
        'face_names': face_names,
        'face_locations': face_locations,
        'annotated_image': annotated_image,
        'detection_time': current_time
    }

def batch_process_images(image_paths, use_database=True):
    """Process multiple images in a loop, store results in session_state."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image_path in enumerate(image_paths):
        status_text.text(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        result = process_single_image(image_path, use_database)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(image_paths))
        time.sleep(0.1)  # let UI refresh
    
    status_text.text("Processing complete!")
    return results

##############################################################################
# Main Streamlit App
##############################################################################
def main():
    st.markdown('<div class="main-header" style="font-size: 2.5rem; font-weight: 700; color: #1E3A8A;">Face Recognition System</div>', unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.subheader("Database Connection")
        
        db_status = "Connected" if st.session_state.db_connected else "Disconnected"
        db_status_color = "green" if st.session_state.db_connected else "orange"
        st.markdown(f"**Status:** <span style='color:{db_status_color}'>{db_status}</span>", unsafe_allow_html=True)
        
        if not st.session_state.db_connected:
            if st.button("Connect to API"):
                with st.spinner("Testing API connectivity..."):
                    connected = test_api_connection()
                    if connected:
                        try:
                            known_ids, known_encs = get_all_known_faces()
                            st.session_state.known_face_ids = known_ids
                            st.session_state.known_face_encodings = [np.array(enc, dtype=np.float64) for enc in known_encs]
                            st.session_state.db_connected = True
                            st.success("Connected successfully!")
                        except Exception as e:
                            st.error(f"Error retrieving known faces: {e}")
                    else:
                        st.error("Could not connect to API. Check logs or server status.")
        else:
            if st.button("Disconnect"):
                st.session_state.db_connected = False
                st.session_state.known_face_ids = []
                st.session_state.known_face_encodings = []
                st.success("Disconnected.")
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio("Select Mode", 
                        ["Single Image", "Batch Processing", "Real-time (Webcam)", "Data Analysis"],
                        index=["Single Image", "Batch Processing", "Real-time (Webcam)", "Data Analysis"].index(st.session_state.mode),
                        key="mode_selection")
        
        # Update session_state so we remember the last mode
        st.session_state.mode = mode
        
        # Display metrics if connected
        if st.session_state.db_connected:
            st.markdown("#### Database Metrics")
            with st.spinner("Loading metrics..."):
                try:
                    num_faces = len(st.session_state.known_face_ids)
                    num_images = get_image_count()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Known Faces", num_faces)
                    col2.metric("Images in DB", num_images)
                except Exception as e:
                    st.error(f"Error loading metrics: {str(e)}")
    
    # --- MAIN PANEL ---
    if mode == "Single Image":
        st.subheader("Single Image Processing")
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        
        if uploaded_file and st.button("Process Image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_file_path = tmp.name
            
            with st.spinner("Processing image..."):
                result = process_single_image(temp_file_path, use_database=use_db)
                if result:
                    # Store in session_state so we can re-display even if mode changes
                    st.session_state.processed_images = [result]
                    st.success(f"Detected {result['face_count']} face(s)")
                    
                    if result['annotated_image'] is not None:
                        st.image(result['annotated_image'], caption="Processed Image", use_container_width=True)
                        
                        st.subheader("Face Details")
                        face_cols = st.columns(min(3, max(1, result['face_count'])))
                        for i, (face_id, face_name, face_loc) in enumerate(
                                zip(result['face_ids'], result['face_names'], result['face_locations'])):
                            col_idx = i % len(face_cols)
                            with face_cols[col_idx]:
                                st.write(f"**{face_name}**")
                                st.write(f"Position: (Top={face_loc[0]}, Right={face_loc[1]}, Bottom={face_loc[2]}, Left={face_loc[3]})")
                                if use_db and isinstance(face_id, int):
                                    try:
                                        details_dict = get_face_details_batch([face_id])
                                        # details_dict = { face_id: {...} }
                                        face_details = details_dict.get(str(face_id)) or details_dict.get(face_id)
                                        if face_details:
                                            st.write(f"First seen: {face_details['first_seen']}")
                                            st.write(f"Last seen: {face_details['last_seen']}")
                                            st.write(f"Appearance count: {face_details['appearance_count']}")
                                        else:
                                            st.write("No details found.")
                                    except Exception as ex:
                                        st.error(f"Error fetching face details: {ex}")
                    else:
                        st.warning("Could not process image or no faces detected.")
                os.unlink(temp_file_path)
    
    elif mode == "Batch Processing":
        st.subheader("Batch Image Processing")
        
        upload_option = st.radio("Select Upload Method", ["Upload Files", "Use Image Folder"])
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        
        if upload_option == "Upload Files":
            uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if uploaded_files and st.button("Process Uploaded Images"):
                temp_paths = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(f.getvalue())
                        temp_paths.append(tmp.name)
                
                with st.spinner("Processing images..."):
                    results = batch_process_images(temp_paths, use_database=use_db)
                    st.session_state.processed_images = results
                
                for path in temp_paths:
                    os.unlink(path)
        
        else:
            folder_path = st.text_input("Enter folder path containing images:")
            if folder_path and st.button("Process Folder"):
                if not os.path.isdir(folder_path):
                    st.error("Invalid folder path")
                else:
                    image_paths = []
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_paths.extend([
                            os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) if f.lower().endswith(ext)
                        ])
                    if not image_paths:
                        st.warning("No images found in the folder.")
                    else:
                        with st.spinner(f"Processing {len(image_paths)} images..."):
                            results = batch_process_images(image_paths, use_database=use_db)
                            st.session_state.processed_images = results
    
    elif mode == "Real-time (Webcam)":
        st.subheader("Real-time Face Recognition")
        st.info("This feature processes frames from your webcam in real-time.")
        
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        
        if st.button("Start Camera"):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam.")
                else:
                    frame_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                    info_placeholder = st.empty()
                    
                    stop_clicked = stop_button_placeholder.button("Stop Camera")
                    
                    # We'll keep a local list of processed frames
                    temp_results = []
                    
                    while not stop_clicked:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Could not read frame.")
                            break
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            cv2.imwrite(tmp.name, frame)
                            temp_file_path = tmp.name
                        
                        result = process_single_image(temp_file_path, use_database=use_db)
                        
                        if result and result['annotated_image'] is not None:
                            frame_placeholder.image(result['annotated_image'], channels="RGB", use_container_width=True)
                            info_text = f"Detected {result['face_count']} face(s): " + ", ".join(result['face_names'])
                            info_placeholder.markdown(info_text)
                            temp_results.append(result)
                        else:
                            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                            info_placeholder.markdown("No faces detected.")
                        
                        os.unlink(temp_file_path)
                        
                        stop_clicked = stop_button_placeholder.button("Stop Camera")
                        time.sleep(0.1)
                    
                    cap.release()
                    st.success("Camera stopped.")
                    # Store the last processed webcam frames to session_state
                    if temp_results:
                        st.session_state.processed_images = temp_results
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif mode == "Data Analysis":
        st.subheader("Face Recognition Analytics")
        
        if not st.session_state.db_connected:
            st.warning("Database connection is required for analytics. Please connect first.")
        else:
            with st.spinner("Loading face data..."):
                try:
                    raw_data = get_face_appearance_data()  # list of dicts
                    if not raw_data:
                        raw_data = []
                except Exception as e:
                    st.error(f"Error fetching face appearance data: {e}")
                    raw_data = []
            
            if raw_data:
                face_data = pd.DataFrame(raw_data)
                tab1, tab2, tab3 = st.tabs(["Face Appearances", "Time Distribution", "Face Comparisons"])
                
                with tab1:
                    st.markdown("### Face Appearance Frequency")
                    face_counts = face_data.groupby('face_id').size().reset_index(name='appearances')
                    face_counts = face_counts.sort_values('appearances', ascending=False)
                    
                    fig = px.bar(
                        face_counts,
                        x='face_id',
                        y='appearances',
                        title='Number of Appearances by Face ID',
                        labels={'face_id': 'Face ID', 'appearances': 'Appearances'},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(face_counts.head(10))
                
                with tab2:
                    st.markdown("### Time Distribution Analysis")
                    if 'detection_time' in face_data.columns:
                        if not pd.api.types.is_datetime64_any_dtype(face_data['detection_time']):
                            face_data['detection_time'] = pd.to_datetime(face_data['detection_time'])
                        
                        face_data['hour'] = face_data['detection_time'].dt.hour
                        face_data['day_of_week'] = face_data['detection_time'].dt.day_name()
                        
                        hour_counts = face_data.groupby('hour').size().reset_index(name='count')
                        fig1 = px.line(hour_counts, x='hour', y='count', markers=True,
                                       title='Face Detections by Hour of Day')
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = face_data.groupby('day_of_week').size().reset_index(name='count')
                        day_counts['day_of_week'] = pd.Categorical(day_counts['day_of_week'], 
                                                                   categories=day_order, 
                                                                   ordered=True)
                        day_counts = day_counts.sort_values('day_of_week')
                        fig2 = px.bar(day_counts, x='day_of_week', y='count',
                                      title='Face Detections by Day of Week')
                        st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.markdown("### Face Comparison Analysis")
                    
                    face_ids = sorted(face_data['face_id'].unique())
                    selected_faces = st.multiselect("Select faces to compare", face_ids, default=face_ids[:min(5, len(face_ids))])
                    
                    if selected_faces:
                        filtered_data = face_data[face_data['face_id'].isin(selected_faces)]
                        
                        if 'detection_time' in filtered_data.columns:
                            filtered_data['detection_time'] = pd.to_datetime(filtered_data['detection_time'])
                            filtered_data['date'] = filtered_data['detection_time'].dt.date
                            face_date_counts = filtered_data.groupby(['face_id', 'date']).size().reset_index(name='appearances')
                            
                            fig_line = px.line(
                                face_date_counts,
                                x='date',
                                y='appearances',
                                color='face_id',
                                title='Face Appearances Over Time',
                                markers=True
                            )
                            st.plotly_chart(fig_line, use_container_width=True)
                            
                            st.markdown("#### Face Co-occurrence Matrix")
                            with st.spinner("Calculating co-occurrences..."):
                                matrix = get_face_co_occurrences(selected_faces)
                            fig_co = px.imshow(
                                matrix,
                                x=selected_faces,
                                y=selected_faces,
                                title='Co-occurrence Matrix',
                                labels=dict(x="Face ID", y="Face ID", color="Frequency")
                            )
                            st.plotly_chart(fig_co, use_container_width=True)
                            
                            st.markdown("#### Time Distribution by Face")
                            dist_option = st.radio("View time distribution for:",
                                                   ["All Selected Faces", "Specific Face"],
                                                   horizontal=True)
                            specific_face_id = None
                            if dist_option == "Specific Face":
                                specific_face_id = st.selectbox("Select a face", selected_faces)
                            
                            with st.spinner("Fetching time distribution..."):
                                dist_data = get_face_time_distribution(face_id=specific_face_id)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                hours = sorted(dist_data['hour_distribution'].keys())
                                hour_counts = [dist_data['hour_distribution'][h] for h in hours]
                                fig_hour = px.bar(
                                    x=hours, 
                                    y=hour_counts, 
                                    labels={'x': 'Hour', 'y': 'Appearances'},
                                    title='Appearances by Hour'
                                )
                                st.plotly_chart(fig_hour, use_container_width=True)
                            
                            with col2:
                                days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                                day_counts = [dist_data['day_distribution'].get(d, 0) for d in days]
                                fig_day = px.bar(
                                    x=days, 
                                    y=day_counts,
                                    labels={'x': 'Day of Week', 'y': 'Appearances'},
                                    title='Appearances by Day of Week'
                                )
                                st.plotly_chart(fig_day, use_container_width=True)
                            
                            st.metric("Total Appearances", dist_data['total_appearances'])
            else:
                st.info("No face data available. Process some images first.")
    
    st.markdown("---")
    st.markdown("### Application Settings")
    face_threshold = st.slider(
        "Face Recognition Threshold", 0.0, 1.0, FACE_DISTANCE_THRESHOLD, 0.01,
        help="Lower values = stricter matching (default ~0.6)."
    )
    
    # Purge old records
    if st.session_state.db_connected:
        st.subheader("Database Maintenance")
        db_retention = st.slider("Retention Period (days)", 1, 365, 90)
        if st.button("Purge Old Records"):
            cutoff_date = (datetime.now() - timedelta(days=db_retention)).strftime("%Y-%m-%d")
            with st.spinner(f"Purging records older than {cutoff_date}..."):
                try:
                    deleted = delete_old_records(cutoff_date)
                    st.success(
                        f"Purged {deleted['image_records']} image records, "
                        f"{deleted['image_counts']} image counts, and "
                        f"{deleted['faces']} faces older than {cutoff_date}"
                    )
                except Exception as e:
                    st.error(f"Error purging records: {e}")
    
    # Export data
    if st.session_state.db_connected:
        st.subheader("Export Data")
        export_format = st.radio("Export Format", ["CSV", "JSON", "Excel"], horizontal=True)
        if st.button("Export"):
            with st.spinner(f"Exporting data as {export_format}..."):
                try:
                    file_bytes = export_data(export_format)
                    if export_format == "CSV":
                        mime, filename = "text/csv", "face_recognition_data.csv"
                    elif export_format == "JSON":
                        mime, filename = "application/json", "face_recognition_data.json"
                    else:
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        filename = "face_recognition_data.xlsx"
                    
                    st.download_button(
                        label=f"Download {export_format}",
                        data=file_bytes,
                        file_name=filename,
                        mime=mime
                    )
                except Exception as e:
                    st.error(f"Error exporting data: {e}")

    # ---------------------------------------------------------------------- #
    # Always show the last processed images, regardless of the current mode.  #
    # This ensures the user can still see them even if they switch modes.     #
    # ---------------------------------------------------------------------- #
    if st.session_state.processed_images:
        st.markdown("## Last Processed Images (Persist across mode changes)")
        # For example, we can do a simple gallery:
        results = st.session_state.processed_images
        st.write(f"Total images in memory: {len(results)}")

        # Example: show only up to 6
        max_show = 6
        displayed = 0
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            if displayed >= max_show:
                break
            cols = st.columns(min(cols_per_row, len(results) - i))
            for j, col in enumerate(cols):
                if i + j < len(results):
                    if displayed >= max_show:
                        break
                    res = results[i + j]
                    caption_text = f"{os.path.basename(res['image_path'])} ({res['face_count']} faces)"
                    if res['annotated_image'] is not None:
                        col.image(res['annotated_image'], caption=caption_text, use_container_width=True)
                    else:
                        col.write(caption_text)
                    displayed += 1


if __name__ == "__main__":
    main()
