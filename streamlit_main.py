import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
import face_recognition
import cv2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
from image_processor import ImageProcessor
from config import FACE_DISTANCE_THRESHOLD
import logging
import io
import base64
from PIL import Image
import time

# Import your database handler - comment this out if you're running in no-db mode
from app.crud.attendance import AsyncDatabaseHandler

##############################################################################
# 1. Enable nest_asyncio and define run_async_task helper
##############################################################################
# nest_asyncio allows us to reuse the existing event loop in Streamlit
nest_asyncio.apply()
# Create a global event loop and set it as the current one.
global_loop = asyncio.new_event_loop()
asyncio.set_event_loop(global_loop)

def run_async_task(coro):
    # Always use the global event loop.
    return global_loop.run_until_complete(coro)
# def run_async_task(coro):
#     """
#     Run an async task in the current event loop (works around 'attached to a 
#     different loop' errors by avoiding asyncio.run in a fresh loop).
#     """
#     loop = asyncio.get_event_loop()
#     return loop.run_until_complete(coro)

# Set page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'known_face_ids' not in st.session_state:
    st.session_state.known_face_ids = []
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'db_handler' not in st.session_state:
    st.session_state.db_handler = None
if 'image_processor' not in st.session_state:
    st.session_state.image_processor = ImageProcessor()
if "db_lock" not in st.session_state:
    st.session_state.db_lock = asyncio.Lock()

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 0.75rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.25rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
    }
    .warning {
        color: #B91C1C;
        font-weight: 600;
    }
    .success {
        color: #15803D;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True
)

##############################################################################
# Asynchronous database helper functions
##############################################################################
async def connect_to_database():
    """Connect to the database and load face data"""
    try:
        db_handler = AsyncDatabaseHandler()
        await db_handler.connect()
        known_face_ids, known_face_encodings = await db_handler.get_all_known_faces()
        
        st.session_state.db_handler = db_handler
        st.session_state.known_face_ids = known_face_ids
        st.session_state.known_face_encodings = known_face_encodings
        st.session_state.db_connected = True
        st.rerun()
        return True
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return False

async def disconnect_from_database():
    """Disconnect from the database"""
    if st.session_state.db_handler:
        await st.session_state.db_handler.close()
        st.session_state.db_handler = None
        st.session_state.db_connected = False

def get_image_with_faces(image_path, face_locations, face_names):
    """Draw rectangles and names on faces in the image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(rgb_image, (left, bottom), (right, bottom + 35), (0, 255, 0), cv2.FILLED)
        cv2.putText(rgb_image, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    return rgb_image

def image_to_base64(image_array):
    """Convert image array to base64 string for display"""
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

async def process_single_image(image_path, use_database=True):
    """Process a single image and return the results"""
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
            distances = face_recognition.face_distance(st.session_state.known_face_encodings, face_encoding)
            min_distance = min(distances) if len(distances) > 0 else 1.0
            
            if min_distance < FACE_DISTANCE_THRESHOLD:
                index = np.argmin(distances)
                face_id = st.session_state.known_face_ids[index]
                if st.session_state.db_handler:
                    await st.session_state.db_handler.update_last_seen(face_id, current_time, face_encoding)
                st.session_state.known_face_encodings[index] = face_encoding
                face_names.append(f"ID: {face_id}")
            else:
                if st.session_state.db_handler:
                    face_id = await st.session_state.db_handler.insert_new_face(face_encoding, current_time)
                    if face_id:
                        st.session_state.known_face_ids.append(face_id)
                        st.session_state.known_face_encodings.append(face_encoding)
                        face_names.append(f"New ID: {face_id}")
        elif use_database and not st.session_state.known_face_encodings:
            if st.session_state.db_handler:
                face_id = await st.session_state.db_handler.insert_new_face(face_encoding, current_time)
                if face_id:
                    st.session_state.known_face_ids.append(face_id)
                    st.session_state.known_face_encodings.append(face_encoding)
                    face_names.append(f"New ID: {face_id}")
        
        if use_database and face_id:
            face_ids_in_image.append(face_id)
            if st.session_state.db_handler:
                await st.session_state.db_handler.insert_image_record(result['image_path'], face_id, current_time)
        else:
            temp_id = f"temp_{hash(face_encoding.tobytes())}"
            face_names.append(f"Temp ID: {temp_id[:8]}")
            face_ids_in_image.append(temp_id)
    
    face_count = len(face_encodings)
    if use_database and st.session_state.db_handler:
        await st.session_state.db_handler.insert_or_update_image_count(result['image_path'], face_count, current_time)
    
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

async def batch_process_images(image_paths, use_database=True):
    """Process multiple images and return the results"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, image_path in enumerate(image_paths):
        status_text.text(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        result = await process_single_image(image_path, use_database)
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(image_paths))
        
        # Non-blocking pause so the loop yields to the event loop
        await asyncio.sleep(0.1)
    
    status_text.text("Processing complete!")
    return results

async def get_face_appearance_data():
    """Get face appearance data from the database for visualization"""
    if not st.session_state.db_connected or not st.session_state.db_handler:
        return None
    
    try:
        face_records = await st.session_state.db_handler.get_face_appearance_data()
        if not face_records:
            return None
            
        df = pd.DataFrame(face_records)
        return df
    except Exception as e:
        st.error(f"Error fetching face appearance data: {str(e)}")
        return None

async def get_face_details(face_id):
    """Get details for a specific face ID"""
    if not st.session_state.db_connected or not st.session_state.db_handler:
        st.write("DEBUG: Database is not connected or db_handler is not set.")
        return None
    
    try:
        st.write(f"DEBUG: Calling db_handler.get_face_details with face_id = {face_id}")
        async with st.session_state.db_lock:
            details = await st.session_state.db_handler.get_face_details(face_id)
            st.write("DEBUG: Retrieved face details:", details)
            return details
    except Exception as e:
        st.error(f"Error fetching face details: {str(e)}")
        st.write("DEBUG: Exception in get_face_details:", e)
        return None

##############################################################################
# Main application UI
##############################################################################
def main():
    st.markdown('<div class="main-header">Face Recognition System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sub-header">Control Panel</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Database Connection")
        
        db_status = "Connected" if st.session_state.db_connected else "Disconnected"
        db_status_color = "success" if st.session_state.db_connected else "warning"
        st.markdown(
            f'<div class="metric-label">Status: <span class="{db_status_color}">{db_status}</span></div>',
            unsafe_allow_html=True
        )
        
        # Use run_async_task(...) instead of asyncio.run(...) to connect/disconnect
        if not st.session_state.db_connected:
            print("Button connect clicked")
            if st.button("Connect to Database"):
                with st.spinner("Connecting to database..."):
                    run_async_task(connect_to_database())
        else:
            print("Button disconnect clicked")
            if st.button("Disconnect"):
                with st.spinner("Disconnecting..."):
                    run_async_task(disconnect_from_database())
                    
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mode selection
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Processing Mode")
        mode = st.radio("Select Mode", ["Single Image", "Batch Processing", "Real-time (Webcam)", "Data Analysis"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display metrics if connected to DB
        if st.session_state.db_connected:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Database Metrics")
            
            with st.spinner("Loading metrics..."):
                try:
                    num_faces = len(st.session_state.known_face_ids)
                    num_images = 0
                    if st.session_state.db_handler:
                        num_images = run_async_task(st.session_state.db_handler.get_image_count())
                    
                    col1, col2 = st.columns(2)
                    col1.markdown(
                        f'<div class="metric-label">Known Faces</div>'
                        f'<div class="metric-value">{num_faces}</div>',
                        unsafe_allow_html=True
                    )
                    col2.markdown(
                        f'<div class="metric-label">Images</div>'
                        f'<div class="metric-value">{num_images}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error loading metrics: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    if mode == "Single Image":
        st.markdown('<div class="sub-header">Single Image Processing</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        process_button = st.button("Process Image")
        
        if uploaded_file and process_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_file_path = tmp.name
            
            with st.spinner("Processing image..."):
                # Use run_async_task to call the async function
                result = run_async_task(process_single_image(temp_file_path, use_database=use_db))
                
                if result:
                    st.success(f"Detected {result['face_count']} face(s) in the image")
                    
                    if result['annotated_image'] is not None:
                        st.image(result['annotated_image'], caption="Processed Image", use_column_width=True)
                        
                        st.markdown('<div class="sub-header">Face Details</div>', unsafe_allow_html=True)
                        face_cols = st.columns(min(3, max(1, result['face_count'])))
                        
                        for i, (face_id, face_name, face_loc) in enumerate(
                                zip(result['face_ids'], result['face_names'], result['face_locations'])):
                            col_idx = i % len(face_cols)
                            with face_cols[col_idx]:
                                st.markdown(f"**{face_name}**")
                                st.markdown(
                                    f"Position: Top={face_loc[0]}, Right={face_loc[1]}, "
                                    f"Bottom={face_loc[2]}, Left={face_loc[3]}"
                                )
                                
                                # Debug info
                                st.write("DEBUG: use_db =", use_db)
                                st.write("DEBUG: st.session_state.db_connected =", st.session_state.db_connected)
                                st.write("DEBUG: face_name =", face_name)
                                st.write("DEBUG: face_id =", face_id)

                                if use_db and st.session_state.db_connected:
                                    st.write("DEBUG: Database conditions met")
                                else:
                                    st.write("DEBUG: Database conditions NOT met")

                                if not face_name.startswith("Temp"):
                                    st.write("DEBUG: face_name does not start with 'Temp'")
                                else:
                                    st.write("DEBUG: face_name starts with 'Temp'")

                                # Only fetch details if we have a real ID
                                if use_db and st.session_state.db_connected and not face_name.startswith("Temp"):
                                    st.write("DEBUG: About to clean face_id and call get_face_details")
                                    face_id_clean = face_id if isinstance(face_id, int) else face_id.replace("temp_", "")
                                    st.write("DEBUG: face_id_clean =", face_id_clean)
                                    
                                    try:
                                        face_details = run_async_task(get_face_details(face_id_clean))
                                        st.write("DEBUG: face_details =", face_details)
                                        if face_details:
                                            st.markdown(f"First seen: {face_details['first_seen']}")
                                            st.markdown(f"Last seen: {face_details['last_seen']}")
                                            st.markdown(f"Total appearances: {face_details['appearance_count']}")
                                        else:
                                            st.write("DEBUG: No face details returned")
                                    except Exception as e:
                                        st.write("DEBUG: Exception when calling get_face_details:", e)
                                else:
                                    st.write(
                                        "DEBUG: Skipping get_face_details block "
                                        "because one or more conditions were not met."
                                    )
                    else:
                        st.warning("Could not process the image or no faces detected.")
                
                os.unlink(temp_file_path)
    
    elif mode == "Batch Processing":
        st.markdown('<div class="sub-header">Batch Image Processing</div>', unsafe_allow_html=True)
        
        upload_option = st.radio("Select upload method", ["Upload Files", "Use Image Folder"])
        
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        
        if upload_option == "Upload Files":
            uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
            if uploaded_files and st.button("Process Uploaded Images"):
                temp_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_paths.append(tmp.name)
                
                with st.spinner("Processing images..."):
                    results = run_async_task(batch_process_images(temp_paths, use_database=use_db))
                    st.session_state.processed_images = results
                
                for path in temp_paths:
                    os.unlink(path)
        
        else:  # Use Image Folder
            folder_path = st.text_input("Enter folder path containing images:")
            
            if folder_path and st.button("Process Folder"):
                if not os.path.isdir(folder_path):
                    st.error("Invalid folder path")
                else:
                    image_paths = []
                    for ext in ['.jpg', '.jpeg', '.png']:
                        image_paths.extend(
                            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                             if f.lower().endswith(ext)]
                        )
                    
                    if not image_paths:
                        st.warning("No images found in the specified folder")
                    else:
                        with st.spinner(f"Processing {len(image_paths)} images..."):
                            results = run_async_task(batch_process_images(image_paths, use_database=use_db))
                            st.session_state.processed_images = results
        
        if st.session_state.processed_images:
            st.markdown('<div class="sub-header">Processing Results</div>', unsafe_allow_html=True)
            st.write(f"Processed {len(st.session_state.processed_images)} images")
            
            total_faces = sum(result['face_count'] for result in st.session_state.processed_images)
            st.markdown(
                f"<div class='metric-label'>Total faces detected: "
                f"<span class='metric-value'>{total_faces}</span></div>",
                unsafe_allow_html=True
            )
            
            cols_per_row = 3
            for i in range(0, len(st.session_state.processed_images), cols_per_row):
                cols = st.columns(min(cols_per_row, len(st.session_state.processed_images) - i))
                
                for j, col in enumerate(cols):
                    if i + j < len(st.session_state.processed_images):
                        result = st.session_state.processed_images[i + j]
                        col.image(
                            result['annotated_image'],
                            caption=f"{os.path.basename(result['image_path'])} ({result['face_count']} faces)",
                            use_column_width=True
                        )
            
            with st.expander("View Detailed Results"):
                for result in st.session_state.processed_images:
                    st.markdown(f"### {os.path.basename(result['image_path'])}")
                    st.markdown(f"Detected {result['face_count']} face(s):")
                    st.markdown(", ".join(result['face_names']))
                    st.image(result['annotated_image'], use_column_width=True)
                    st.markdown("---")
    
    elif mode == "Real-time (Webcam)":
        st.markdown('<div class="sub-header">Real-time Face Recognition</div>', unsafe_allow_html=True)
        
        st.info("This feature processes video from your webcam to recognize faces in real-time.")
        
        use_db = st.checkbox("Use Database", value=st.session_state.db_connected, disabled=not st.session_state.db_connected)
        
        if st.button("Start Camera"):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not open webcam.")
                else:
                    frame_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                    info_placeholder = st.empty()
                    
                    stop_clicked = stop_button_placeholder.button("Stop Camera")
                    
                    while not stop_clicked:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error: Could not read frame.")
                            break
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            cv2.imwrite(tmp.name, frame)
                            temp_file_path = tmp.name
                        
                        # Use run_async_task here too
                        result = run_async_task(process_single_image(temp_file_path, use_database=use_db))
                        
                        if result and result['annotated_image'] is not None:
                            frame_placeholder.image(result['annotated_image'], channels="RGB", use_column_width=True)
                            info_text = f"Detected {result['face_count']} face(s): " + ", ".join(result['face_names'])
                            info_placeholder.markdown(info_text)
                        else:
                            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                            info_placeholder.markdown("No faces detected")
                        
                        os.unlink(temp_file_path)
                        
                        stop_clicked = stop_button_placeholder.button("Stop Camera")
                        time.sleep(0.1)
                    
                    cap.release()
                    st.success("Camera stopped")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif mode == "Data Analysis":
        st.markdown('<div class="sub-header">Face Recognition Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.db_connected:
            st.warning("Database connection required for analytics. Please connect to the database first.")
        else:
            with st.spinner("Loading face data..."):
                # run_async_task for retrieving face data
                face_data = run_async_task(get_face_appearance_data())
            
            if face_data is not None and not face_data.empty:
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
                        labels={'face_id': 'Face ID', 'appearances': 'Number of Appearances'},
                        color='appearances',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### Top Faces")
                    st.dataframe(face_counts.head(10))
                
                with tab2:
                    st.markdown("### Time Distribution Analysis")
                    
                    if 'detection_time' in face_data.columns:
                        if not pd.api.types.is_datetime64_any_dtype(face_data['detection_time']):
                            face_data['detection_time'] = pd.to_datetime(face_data['detection_time'])
                        
                        face_data['hour'] = face_data['detection_time'].dt.hour
                        face_data['day_of_week'] = face_data['detection_time'].dt.day_name()
                        
                        hour_counts = face_data.groupby('hour').size().reset_index(name='count')
                        
                        fig1 = px.line(
                            hour_counts,
                            x='hour',
                            y='count',
                            title='Face Detections by Hour of Day',
                            labels={'hour': 'Hour of Day (24h)', 'count': 'Number of Detections'},
                            markers=True
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_counts = face_data.groupby('day_of_week').size().reset_index(name='count')
                        day_counts['day_of_week'] = pd.Categorical(day_counts['day_of_week'], categories=day_order, ordered=True)
                        day_counts = day_counts.sort_values('day_of_week')
                        
                        fig2 = px.bar(
                            day_counts,
                            x='day_of_week',
                            y='count',
                            title='Face Detections by Day of Week',
                            labels={'day_of_week': 'Day of Week', 'count': 'Number of Detections'},
                            color='count',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    st.markdown("### Face Comparison Analysis")
                    
                    face_ids = sorted(face_data['face_id'].unique())
                    selected_faces = st.multiselect("Select faces to compare", face_ids, default=face_ids[:min(5, len(face_ids))])
                    
                    if selected_faces:
                        filtered_data = face_data[face_data['face_id'].isin(selected_faces)]
                        
                        if 'detection_time' in filtered_data.columns:
                            filtered_data['date'] = filtered_data['detection_time'].dt.date
                            face_date_counts = filtered_data.groupby(['face_id', 'date']).size().reset_index(name='appearances')
                            
                            fig = px.line(
                                face_date_counts,
                                x='date',
                                y='appearances',
                                color='face_id',
                                title='Face Appearances Over Time',
                                labels={'date': 'Date', 'appearances': 'Number of Appearances', 'face_id': 'Face ID'},
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### Face Co-occurrence")
                            st.markdown("This heatmap shows how often different faces appear together in the same images.")
                            
                            with st.spinner("Calculating co-occurrences..."):
                                correlation_matrix = run_async_task(
                                    st.session_state.db_handler.get_face_co_occurrences(selected_faces)
                                )
                            
                            fig = px.imshow(
                                correlation_matrix,
                                x=selected_faces,
                                y=selected_faces,
                                color_continuous_scale='Blues',
                                title='Face Co-occurrence Matrix',
                                labels=dict(x="Face ID", y="Face ID", color="Frequency")
                            )
                            fig.update_layout(
                                xaxis_title='Face ID',
                                yaxis_title='Face ID'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### Time Distribution by Face")
                            dist_option = st.radio(
                                "View time distribution for:",
                                ["All Selected Faces", "Specific Face"],
                                horizontal=True
                            )
                            
                            specific_face_id = None
                            if dist_option == "Specific Face":
                                specific_face_id = st.selectbox("Select a face", selected_faces)
                            
                            with st.spinner("Calculating time distribution..."):
                                time_dist = run_async_task(
                                    st.session_state.db_handler.get_face_time_distribution(
                                        face_id=specific_face_id if dist_option == "Specific Face" else None
                                    )
                                )
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                hours = list(range(24))
                                hour_counts = [time_dist['hour_distribution'].get(hour, 0) for hour in hours]
                                
                                fig_hour = px.bar(
                                    x=hours,
                                    y=hour_counts,
                                    labels={'x': 'Hour of Day', 'y': 'Number of Appearances'},
                                    title='Appearances by Hour of Day'
                                )
                                fig_hour.update_layout(bargap=0.1)
                                st.plotly_chart(fig_hour, use_container_width=True)
                            
                            with col2:
                                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                day_counts = [time_dist['day_distribution'].get(day, 0) for day in days]
                                
                                fig_day = px.bar(
                                    x=days,
                                    y=day_counts,
                                    labels={'x': 'Day of Week', 'y': 'Number of Appearances'},
                                    title='Appearances by Day of Week'
                                )
                                fig_day.update_layout(bargap=0.2)
                                st.plotly_chart(fig_day, use_container_width=True)
                            
                            st.metric("Total Appearances", time_dist['total_appearances'])
            else:
                st.info("No face data available. Process some images first to generate analytics.")

    st.markdown("---")
    st.markdown("### Face Recognition System")
    st.markdown("Built with Streamlit and face_recognition library")
    
    with st.expander("Settings"):
        st.subheader("Application Settings")
        
        # Face detection settings
        st.markdown("#### Face Detection Parameters")
        face_threshold = st.slider(
            "Face Recognition Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=FACE_DISTANCE_THRESHOLD,
            step=0.01,
            help="Lower values are more strict (require closer match). Default is 0.6."
        )
        
        # Image processing settings
        st.markdown("#### Image Processing Settings")
        img_size = st.select_slider(
            "Image Processing Size",
            options=["Small", "Medium", "Large", "Original"],
            value="Medium",
            help="Smaller sizes process faster but may miss smaller faces."
        )
        
        # Database settings (if connected)
        if st.session_state.db_connected:
            st.markdown("#### Database Settings")
            db_retention = st.slider(
                "Database Retention Period (days)",
                min_value=1,
                max_value=365,
                value=90,
                step=1,
                help="Images older than this will be automatically purged."
            )
            
            if st.button("Purge Old Records"):
                cutoff_date = datetime.now() - timedelta(days=db_retention)
                
                with st.spinner(f"Purging records older than {cutoff_date.strftime('%Y-%m-%d')}..."):
                    deleted = run_async_task(st.session_state.db_handler.delete_old_records(cutoff_date))
                    
                    st.success(
                        f"Purged {deleted['image_records']} image records, "
                        f"{deleted['image_counts']} image counts, and "
                        f"{deleted['faces']} faces older than {cutoff_date.strftime('%Y-%m-%d')}"
                    )
        
        # Export options
        st.markdown("#### Export Options")
        export_format = st.radio(
            "Export Format",
            ["CSV", "JSON", "Excel"],
            horizontal=True
        )
        
        if st.button("Export Data"):
            if st.session_state.db_connected:
                with st.spinner(f"Exporting data as {export_format}..."):
                    exported_data = run_async_task(
                        st.session_state.db_handler.export_data(format_type=export_format)
                    )
                    
                    if export_format == "CSV":
                        filename = "face_recognition_data.csv"
                        mime = "text/csv"
                    elif export_format == "JSON":
                        filename = "face_recognition_data.json"
                        mime = "application/json"
                    else:  # Excel
                        filename = "face_recognition_data.xlsx"
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    
                    st.success(f"Data exported in {export_format} format.")
                    st.download_button(
                        label=f"Download {export_format} File",
                        data=exported_data,
                        file_name=filename,
                        mime=mime
                    )
            else:
                st.warning("Database connection required for data export.")

if __name__ == "__main__":
    main()
