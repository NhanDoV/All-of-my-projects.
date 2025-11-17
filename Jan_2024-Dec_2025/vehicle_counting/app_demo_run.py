import streamlit as st
import os
from ultralytics import YOLO
import cv2
import pandas as pd
import subprocess
from tqdm import tqdm
from libs.helper import resize_frame  # import helper functions

# Load the trained YOLO model
model_path = "/kaggle/working/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Video input path
video_path = "/kaggle/input/my-videos/2024_0405_181503_062.mp4"

# Configuration parameters
verbose = False
scale_percent = 50
class_IDS = [0,1,2,3,4,5,6]

# Initialize video capture
video = cv2.VideoCapture(video_path)

# Get original video properties
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
print(f"[INFO] - Original Dimensions: {(width, height)} with fps: {fps}")

# Adjust dimensions for scaling if necessary
if scale_percent != 100:
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print(f"[INFO] - Scaled Dimensions: {(width, height)}")

# Prepare video writer for output
output_file = "result.mp4"
video_writer = cv2.VideoWriter(
    output_file,
    cv2.VideoWriter_fourcc(*"MP4V"),
    fps,
    (width, height)
)

# DataFrame to store all detections
all_detections = pd.DataFrame()

# Process video frames (e.g., every 30 frames)
for i in tqdm(range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 30)):
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame using helper function
    frame = resize_frame(frame, scale_percent)

    # Make predictions with YOLO
    y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device=0, verbose=False)
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

    # Add metadata for frame and center points
    positions_frame['frame_id'] = i
    positions_frame['center_y'] = (positions_frame['ymin'] + positions_frame['ymax']) / 2
    positions_frame['center_x'] = (positions_frame['xmin'] + positions_frame['xmax']) / 2

    # Append detections
    all_detections = pd.concat([all_detections, positions_frame])

    # Placeholder for visualization and counting logic

    # Write processed frame to output video
    video_writer.write(frame)

# Release video resources
video_writer.release()
video.release()

# Optional: Post-processing with ffmpeg can be added here

