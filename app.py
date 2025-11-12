import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from collections import defaultdict
import pandas as pd

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

st.title("🏎️ Pit Stop Analysis System")
st.markdown("Upload an overhead video of a pit stop to generate detailed statistics")

@st.cache_resource
def load_model():
    """Load YOLOv8 model for vehicle detection"""
    model = YOLO('yolov8n.pt')  # Using nano model for speed
    return model

def analyze_pit_stop(video_path, model):
    """Analyze pit stop video and extract statistics"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Statistics tracking
    car_positions = []
    stationary_frames = 0
    motion_detected = False
    stop_start_frame = None
    stop_end_frame = None
    crew_count_per_frame = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    prev_car_center = None
    motion_threshold = 10  # pixels
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection every 5 frames for efficiency
        if frame_idx % 5 == 0:
            results = model(frame, classes=[2, 7], verbose=False)  # car=2, truck=7
            
            car_detected = False
            crew_members = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:
                        if cls in [2, 7]:  # Car or truck
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            car_detected = True
                            car_positions.append((frame_idx, center_x, center_y))
                            
                            # Check if car is stationary
                            if prev_car_center is not None:
                                distance = np.sqrt((center_x - prev_car_center[0])**2 + 
                                                 (center_y - prev_car_center[1])**2)
                                if distance < motion_threshold:
                                    stationary_frames += 1
                                    if not motion_detected and stationary_frames > fps * 0.5:
                                        stop_start_frame = frame_idx
                                        motion_detected = True
                                else:
                                    if motion_detected and stationary_frames > fps * 0.5:
                                        stop_end_frame = frame_idx
                            
                            prev_car_center = (center_x, center_y)
                        
                        # Count people (class 0) as crew members
                        elif cls == 0:
                            crew_members += 1
            
            crew_count_per_frame.append(crew_members)
        
        frame_idx += 1
        progress_bar.progress(frame_idx / frame_count)
        status_text.text(f"Processing frame {frame_idx}/{frame_count}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Calculate statistics
    stats = {
        'total_frames': frame_count,
        'fps': fps,
        'duration': duration,
        'car_positions': car_positions,
        'stop_start_frame': stop_start_frame,
        'stop_end_frame': stop_end_frame,
        'avg_crew_count': np.mean(crew_count_per_frame) if crew_count_per_frame else 0,
        'max_crew_count': max(crew_count_per_frame) if crew_count_per_frame else 0
    }
    
    return stats

def display_statistics(stats):
    """Display pit stop statistics"""
    if stats is None:
        return
    
    st.header("📊 Pit Stop Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Video Duration", f"{stats['duration']:.2f}s")
    
    with col2:
        if stats['stop_start_frame'] and stats['stop_end_frame']:
            pit_duration = (stats['stop_end_frame'] - stats['stop_start_frame']) / stats['fps']
            st.metric("Pit Stop Duration", f"{pit_duration:.2f}s")
        else:
            st.metric("Pit Stop Duration", "N/A")
    
    with col3:
        st.metric("Avg Crew Members", f"{stats['avg_crew_count']:.1f}")
    
    with col4:
        st.metric("Max Crew Members", f"{stats['max_crew_count']}")
    
    # Detailed breakdown
    st.subheader("Detailed Analysis")
    
    if stats['stop_start_frame'] and stats['stop_end_frame']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Entry Time:**")
            entry_time = stats['stop_start_frame'] / stats['fps']
            st.write(f"{entry_time:.2f}s into video")
        
        with col2:
            st.write("**Exit Time:**")
            exit_time = stats['stop_end_frame'] / stats['fps']
            st.write(f"{exit_time:.2f}s into video")
    
    # Create DataFrame for detailed view
    if len(stats['car_positions']) > 0:
        st.subheader("Car Position Tracking")
        positions_df = pd.DataFrame(
            stats['car_positions'], 
            columns=['Frame', 'Center X', 'Center Y']
        )
        positions_df['Time (s)'] = positions_df['Frame'] / stats['fps']
        st.dataframe(positions_df.head(20), use_container_width=True)
        
        # Download button for full data
        csv = positions_df.to_csv(index=False)
        st.download_button(
            label="Download Full Position Data",
            data=csv,
            file_name="pit_stop_positions.csv",
            mime="text/csv"
        )

def main():
    st.sidebar.header("About")
    st.sidebar.info(
        "This application analyzes overhead pit stop videos using YOLOv8 object detection. "
        "It tracks the race car's position, detects when it stops, measures pit stop duration, "
        "and counts crew members."
    )
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload an overhead video of a pit stop
    2. Wait for processing to complete
    3. View statistics and insights
    4. Download detailed data if needed
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Pit Stop Video", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload an overhead video showing a pit stop"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.video(tmp_path)
        
        if st.button("Analyze Pit Stop", type="primary"):
            with st.spinner("Loading AI model..."):
                model = load_model()
            
            with st.spinner("Analyzing video... This may take a few minutes."):
                stats = analyze_pit_stop(tmp_path, model)
                
                if stats:
                    st.success("Analysis complete!")
                    display_statistics(stats)
                else:
                    st.error("Failed to analyze video. Please try another file.")

if __name__ == "__main__":
    main()
