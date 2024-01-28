# Accident-Detection-yolov8-streamlit

🚗📹 Real-time Accident Detection using YOLOv8 and Streamlit 📹🚗

## Overview

This project utilizes YOLOv8 (You Only Look Once) object detection model combined with Streamlit to perform real-time accident detection in videos. The application provides a user-friendly interface to upload videos, detect objects, and visualize the detection results.

## Features

📹 **Real-time Object Detection**: Detects and tracks objects in real-time using the YOLOv8 model.

🎥 **Video Analysis**: Supports both uploaded videos and RTSP streams for analysis.

🖼️ **Snapshot Capture**: Saves snapshots of detected frames for further analysis.

📊 **Results Visualization**: Displays detection results with timestamp and video details.

## Prerequisites

- Python 3.x
- Required Python packages: `streamlit`, `opencv-python`, `tinydb`, `ultralytics`
- YOLOv8 model weights (not included, download from [YOLO website](https://github.com/ultralytics/yolov5/releases))

## Usage

1. Install the required Python packages:

    ```bash
    pip install streamlit opencv-python tinydb
    ```

2. Download YOLOv8 model weights from the [YOLOv5 GitHub releases](https://github.com/ultralytics/yolov5/releases).

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

4. Open your web browser and navigate to the provided URL.

## Folder Structure

- 📁 **database**: Stores detection results in JSON format.
- 📁 **snapshots**: Contains captured snapshots of detected frames.
