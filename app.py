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

def find_car_hsv(frame):
    """Find the race car using color detection and return its center position"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for the car (adjust these based on car colors)
    # Green/lime colors (for the car in the video)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # Find the largest contour (likely the car)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter out small detections
    if area < 500:
        return None, mask
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w/2
    center_y = y + h/2
    
    return (center_x, center_y, area), mask

def analyze_pit_stop(video_path):
    """Analyze pit stop by tracking car's horizontal position"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Statistics tracking
    car_positions = []
    horizontal_velocities = []
    stationary_frames = 0
    stop_detected = False
    stop_start_frame = None
    stop_end_frame = None
    crew_activity = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Display sample frames
    sample_col1, sample_col2 = st.columns(2)
    
    frame_idx = 0
    prev_position = None
    prev_gray = None
    
    # Define pit box region for crew detection
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pit_x1, pit_y1 = int(width * 0.075), int(height * 0.075)
    pit_x2, pit_y2 = int(width * 0.925), int(height * 0.925)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find the car
        car_data, mask = find_car_hsv(frame)
        
        if car_data:
            center_x, center_y, area = car_data
            car_positions.append((frame_idx, center_x, center_y, area))
            
            # Calculate horizontal velocity
            if prev_position is not None:
                h_velocity = abs(center_x - prev_position[0])
                horizontal_velocities.append(h_velocity)
                
                # Car is stationary if horizontal movement is less than 2 pixels
                is_stationary = h_velocity < 2.0
                
                if is_stationary and frame_idx > fps * 2:  # Skip first 2 seconds
                    stationary_frames += 1
                    
                    # Detect stop start
                    if not stop_detected and stationary_frames > fps * 1.0:
                        stop_start_frame = frame_idx - int(fps * 1.0)
                        stop_detected = True
                else:
                    # Car is moving
                    if stop_detected and stop_end_frame is None and stationary_frames > fps * 2.0:
                        # Car was stopped for at least 2 seconds and is now moving
                        stop_end_frame = frame_idx
                    
                    if not stop_detected:
                        stationary_frames = 0
            
            prev_position = (center_x, center_y)
        else:
            horizontal_velocities.append(0)
        
        # Detect crew activity using motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            pit_region = thresh[pit_y1:pit_y2, pit_x1:pit_x2]
            edge_region_top = thresh[pit_y1:pit_y1+50, pit_x1:pit_x2]
            edge_region_bottom = thresh[pit_y2-50:pit_y2, pit_x1:pit_x2]
            edge_region_left = thresh[pit_y1:pit_y2, pit_x1:pit_x1+50]
            edge_region_right = thresh[pit_y1:pit_y2, pit_x2-50:pit_x2]
            
            crew_motion = (cv2.countNonZero(edge_region_top) + 
                          cv2.countNonZero(edge_region_bottom) +
                          cv2.countNonZero(edge_region_left) + 
                          cv2.countNonZero(edge_region_right))
            
            crew_activity.append(crew_motion)
        else:
            crew_activity.append(0)
        
        prev_gray = gray
        
        # Show sample frames
        if frame_idx == int(frame_count * 0.3):
            with sample_col1:
                vis_frame = frame.copy()
                if car_data:
                    cv2.circle(vis_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), -1)
                st.image(vis_frame, caption=f"Frame at 30% ({frame_idx}) - Car Detection", use_container_width=True)
                
        if frame_idx == int(frame_count * 0.6):
            with sample_col2:
                vis_frame = frame.copy()
                if car_data:
                    cv2.circle(vis_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), -1)
                cv2.rectangle(vis_frame, (pit_x1, pit_y1), (pit_x2, pit_y2), (0, 255, 0), 3)
                st.image(vis_frame, caption=f"Frame at 60% ({frame_idx}) - Detection Region", use_container_width=True)
        
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / frame_count, 1.0))
            status_text.text(f"Processing frame {frame_idx}/{frame_count}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    # Calculate crew statistics
    if len(crew_activity) > 0:
        crew_activity_array = np.array(crew_activity)
        avg_activity = np.mean(crew_activity_array[crew_activity_array > np.percentile(crew_activity_array, 50)])
        estimated_crew = min(int(avg_activity / 5000), 20)
    else:
        estimated_crew = 0
    
    # Find peak activity
    max_activity_idx = np.argmax(crew_activity) if len(crew_activity) > 0 else 0
    max_activity_time = max_activity_idx / fps if fps > 0 else 0
    
    stats = {
        'total_frames': frame_count,
        'fps': fps,
        'duration': duration,
        'stop_start_frame': stop_start_frame,
        'stop_end_frame': stop_end_frame,
        'car_positions': car_positions,
        'horizontal_velocities': horizontal_velocities,
        'crew_activity': crew_activity,
        'estimated_crew': estimated_crew,
        'max_activity_time': max_activity_time
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
            st.metric("Pit Stop Duration", "Not detected")
    
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
    
    # Car tracking chart
    if len(stats['car_positions']) > 0:
        st.subheader("Car Position Tracking")
        
        positions_df = pd.DataFrame(
            stats['car_positions'], 
            columns=['Frame', 'X Position', 'Y Position', 'Area']
        )
        positions_df['Time (s)'] = positions_df['Frame'] / stats['fps']
        
        # Create chart showing horizontal position and velocity
        chart_data = pd.DataFrame({
            'Time (s)': positions_df['Time (s)'],
            'X Position': positions_df['X Position'],
        })
        
        st.line_chart(chart_data.set_index('Time (s)'))
        
        # Velocity chart
        if len(stats['horizontal_velocities']) > 0:
            st.subheader("Horizontal Velocity")
            velocity_df = pd.DataFrame({
                'Time (s)': [i / stats['fps'] for i in range(len(stats['horizontal_velocities']))],
                'Velocity (pixels/frame)': stats['horizontal_velocities']
            })
            st.line_chart(velocity_df.set_index('Time (s)'))
        
        # Download button
        csv = positions_df.to_csv(index=False)
        st.download_button(
            label="Download Position Data",
            data=csv,
            file_name="pit_stop_car_tracking.csv",
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
        
        if len(stats['car_positions']) > 0:
            insights.append(f"🎯 Car successfully tracked through {len(stats['car_positions'])} frames")
        
        for insight in insights:
            st.info(insight)
    else:
        st.warning("⚠️ Could not detect car stop/start times. Try adjusting the video quality or camera angle.")

def main():
    st.sidebar.header("About")
    st.sidebar.info(
        "This application analyzes overhead pit stop videos by tracking the race car's position. "
        "It detects when the car stops and departs based on horizontal movement."
    )
    
    st.sidebar.header("How It Works")
    st.sidebar.markdown("""
    1. **Car Detection**: Uses HSV color detection to find the car
    2. **Position Tracking**: Monitors horizontal movement
    3. **Stop Detection**: Identifies when car becomes stationary (<2 pixels/frame)
    4. **Departure Detection**: Detects when car starts moving again
    5. **Crew Activity**: Monitors motion around pit box edges
    """)
    
    st.sidebar.header("Tips")
    st.sidebar.markdown("""
    - Works best with overhead camera angles
    - Car should be distinctly colored (green works well)
    - Good lighting improves detection accuracy
    - May need to adjust color thresholds for different cars
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
                stats = analyze_pit_stop(tmp_path)
                
                if stats:
                    st.success("✅ Analysis complete!")
                    display_statistics(stats)
                else:
                    st.error("Failed to analyze video. Please try another file.")

if __name__ == "__main__":
    main()
