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
- Required Python packages: `streamlit`, `opencv-python`, `tinydb`, `ultralytics`, `python-dotenv`, `secure-smtplib`
- Finetuned YOLOv8 model weights (are included or download from [YOLO website](https://github.com/ultralytics/yolov5/releases))

## Demo

### Home page

<img src="https://github.com/000jd/Accident-Detection-yolov8-streamlit/blob/main/demo/Accident-Detection.png" >

### Result Page 

<img src="https://github.com/000jd/Accident-Detection-yolov8-streamlit/blob/main/demo/Accident-Detection%20(1).png" >

## Usage

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and navigate to the provided URL.

## Folder Structure

- 📁 **app.py**: Main application file.
- 📁 **database**: Stores detection results in JSON format.
  - 📄 data.json: JSON file containing detection results.
- 📁 **demo**: Contains demonstration images.
  - 🖼️ **Accident-Detection (1).png**: Demo image 1.
  - 🖼️ **Accident-Detection.png**: Demo image 2.
- 📁 **requirements.txt**: Text file listing project dependencies.
- 📁 **snapshots**: Contains captured snapshots of detected frames.
- 📁 **utils**: Contains utility files.
  - 📁 **images**: Contains additional images for testing.
    - 🖼️ **test_detat.jpg**: Test image.
    - 🖼️ **test.jpg**: Test image.
  - 📄 **helper.py**: Helper functions module.
  - 📄 **notifiction.py**: Notification functions module.
  - 📄 **settings.py**: Settings module containing configurations.
  - 📁 **weights**: Contains pre-trained model weights.

