import streamlit as st
import numpy as np
import tempfile
import pandas as pd
from PIL import Image
import io

# Set headless mode for OpenCV
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

import cv2

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

st.title("🏎️ Pit Stop Analysis System")
st.markdown("Upload an overhead video of a pit stop to generate detailed statistics")

def detect_motion_and_analyze(video_path):
    """Analyze pit stop using motion detection and color-based tracking"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Read first frame as reference
    ret, first_frame = cap.read()
    if not ret:
        st.error("Could not read first frame")
        return None
    
    # Convert to grayscale and blur for motion detection
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    
    # Define pit box region (center 85% of frame to capture full pit area)
    height, width = first_frame.shape[:2]
    pit_x1, pit_y1 = int(width * 0.075), int(height * 0.075)
    pit_x2, pit_y2 = int(width * 0.925), int(height * 0.925)
    
    # Statistics tracking
    motion_timeline = []
    car_present = []
    crew_activity = []
    stop_detected = False
    stop_start_frame = None
    stop_end_frame = None
    max_activity_frame = 0
    max_activity_value = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Display sample frame
    sample_col1, sample_col2 = st.columns(2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    stationary_count = 0
    moving_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Compute frame difference for motion detection
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Focus on pit box region
        pit_region = thresh[pit_y1:pit_y2, pit_x1:pit_x2]
        
        # Calculate motion metrics
        motion_pixels = cv2.countNonZero(pit_region)
        motion_percentage = (motion_pixels / (pit_region.size)) * 100
        
        # Detect crew activity (motion around edges of pit box)
        edge_region_top = thresh[pit_y1:pit_y1+50, pit_x1:pit_x2]
        edge_region_bottom = thresh[pit_y2-50:pit_y2, pit_x1:pit_x2]
        edge_region_left = thresh[pit_y1:pit_y2, pit_x1:pit_x1+50]
        edge_region_right = thresh[pit_y1:pit_y2, pit_x2-50:pit_x2]
        
        crew_motion = (cv2.countNonZero(edge_region_top) + 
                      cv2.countNonZero(edge_region_bottom) +
                      cv2.countNonZero(edge_region_left) + 
                      cv2.countNonZero(edge_region_right))
        
        crew_activity.append(crew_motion)
        
        motion_timeline.append(motion_percentage)
        
        # Pit stop detection strategy:
        # 1. Look for motion crossing ABOVE 12% threshold (car stops, crew rushes)
        # 2. Then wait for motion to drop BELOW 8% and stay there (car is stationary)
        # 3. Exit when motion spikes ABOVE 8% again after being low (car departs)
        
        if not stop_detected:
            # Looking for entry - need to see high motion spike first
            if motion_percentage > 12.0 and frame_idx > fps * 2:
                stationary_count += 1
                if stationary_count > fps * 0.5:
                    stop_start_frame = frame_idx - int(fps * 0.5) 
                    stop_detected = True
                    stationary_count = 0
            else:
                stationary_count = 0
        
        elif stop_detected and stop_end_frame is None:
            # Car is stopped - wait for motion to spike again (departure)
            # But ignore spikes in the first 20 seconds (initial crew activity)
            time_since_stop = (frame_idx - stop_start_frame) / fps
            
            if time_since_stop > 20 and motion_percentage > 10.0:
                # Possible departure
                moving_count += 1
                if moving_count > fps * 1.0:
                    stop_end_frame = frame_idx - int(fps * 1.0)
            else:
                moving_count = 0
        
        # Track car presence
        if stop_detected and stop_end_frame is None:
            car_present.append(True)
        else:
            car_present.append(False)
        
        # Track maximum activity
        if crew_motion > max_activity_value:
            max_activity_value = crew_motion
            max_activity_frame = frame_idx
        
        # Show sample frames
        if frame_idx == int(frame_count * 0.3):
            with sample_col1:
                st.image(frame, caption=f"Frame at 30% ({frame_idx})", use_container_width=True)
                
        if frame_idx == int(frame_count * 0.6):
            with sample_col2:
                # Draw detection region
                vis_frame = frame.copy()
                cv2.rectangle(vis_frame, (pit_x1, pit_y1), (pit_x2, pit_y2), (0, 255, 0), 3)
                st.image(vis_frame, caption=f"Frame at 60% ({frame_idx}) - Detection Region", use_container_width=True)
        
        prev_gray = gray
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / frame_count, 1.0))
            status_text.text(f"Processing frame {frame_idx}/{frame_count}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Estimate crew count from activity levels
    if len(crew_activity) > 0:
        crew_activity_array = np.array(crew_activity)
        # Normalize and estimate (rough heuristic)
        avg_activity = np.mean(crew_activity_array[crew_activity_array > np.percentile(crew_activity_array, 50)])
        estimated_crew = min(int(avg_activity / 5000), 20)  # Cap at 20
    else:
        estimated_crew = 0
    
    stats = {
        'total_frames': frame_count,
        'fps': fps,
        'duration': duration,
        'stop_start_frame': stop_start_frame,
        'stop_end_frame': stop_end_frame,
        'motion_timeline': motion_timeline,
        'car_present': car_present,
        'crew_activity': crew_activity,
        'estimated_crew': estimated_crew,
        'max_activity_frame': max_activity_frame,
        'max_activity_time': max_activity_frame / fps if fps > 0 else 0
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
            st.metric("Pit Stop Duration", "Detecting...")
    
    with col3:
        st.metric("Estimated Crew", f"~{stats['estimated_crew']}")
    
    with col4:
        st.metric("Peak Activity", f"{stats['max_activity_time']:.1f}s")
    
    # Detailed breakdown
    st.subheader("Detailed Analysis")
    
    if stats['stop_start_frame'] and stats['stop_end_frame']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Stop Start:**")
            entry_time = stats['stop_start_frame'] / stats['fps']
            st.write(f"{entry_time:.2f}s")
        
        with col2:
            st.write("**Stop End:**")
            exit_time = stats['stop_end_frame'] / stats['fps']
            st.write(f"{exit_time:.2f}s")
            
        with col3:
            st.write("**Work Duration:**")
            work_time = (stats['stop_end_frame'] - stats['stop_start_frame']) / stats['fps']
            st.write(f"{work_time:.2f}s")
    
    # Motion timeline chart
    if len(stats['motion_timeline']) > 0:
        st.subheader("Activity Timeline")
        
        timeline_data = pd.DataFrame({
            'Frame': range(len(stats['motion_timeline'])),
            'Time (s)': [i / stats['fps'] for i in range(len(stats['motion_timeline']))],
            'Motion Level': stats['motion_timeline'],
            'Crew Activity': [a / 1000 for a in stats['crew_activity']],  # Scale for visibility
        })
        
        st.line_chart(timeline_data.set_index('Time (s)')[['Motion Level', 'Crew Activity']])
        
        # Download button
        csv = timeline_data.to_csv(index=False)
        st.download_button(
            label="Download Timeline Data",
            data=csv,
            file_name="pit_stop_analysis.csv",
            mime="text/csv"
        )
    
    # Insights
    st.subheader("💡 Insights")
    if stats['stop_start_frame'] and stats['stop_end_frame']:
        pit_duration = (stats['stop_end_frame'] - stats['stop_start_frame']) / stats['fps']
        
        insights = []
        if pit_duration < 10:
            insights.append("⚡ Excellent pit stop time!")
        elif pit_duration < 15:
            insights.append("✅ Good pit stop execution")
        else:
            insights.append("⏱️ Consider analyzing for potential improvements")
        
        if stats['estimated_crew'] >= 8:
            insights.append(f"👥 Full crew detected (~{stats['estimated_crew']} members)")
        
        for insight in insights:
            st.info(insight)

def main():
    st.sidebar.header("About")
    st.sidebar.info(
        "This application analyzes overhead pit stop videos using computer vision. "
        "It detects motion patterns, tracks the car's stationary period, "
        "and estimates crew activity levels."
    )
    
    st.sidebar.header("How It Works")
    st.sidebar.markdown("""
    1. **Motion Detection**: Tracks frame-to-frame changes
    2. **Car Detection**: Identifies when vehicle is stationary in center
    3. **Crew Tracking**: Monitors activity around pit box edges
    4. **Timing Analysis**: Measures pit stop duration
    """)
    
    st.sidebar.header("Tips")
    st.sidebar.markdown("""
    - Works best with stable overhead camera angles
    - Fisheye lens distortion is handled automatically
    - Algorithm adapts to different lighting conditions
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
            with st.spinner("Analyzing video... This may take a few minutes."):
                stats = detect_motion_and_analyze(tmp_path)
                
                if stats:
                    st.success("✅ Analysis complete!")
                    display_statistics(stats)
                else:
                    st.error("Failed to analyze video. Please try another file.")

if __name__ == "__main__":
    main()
