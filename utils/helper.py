from ultralytics import YOLO
import time
import streamlit as st
import cv2
from tinydb import TinyDB, Query
import os
from datetime import datetime
import utils.settings as settings
import utils.notifiction as sos

db = TinyDB(settings.DATABASE)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    model.classes = ["Accident"]
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                caption='Detected Video',
                channels="BGR",
                use_column_width=True
                )

def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                            model,
                                            st_frame,
                                            image,
                                            is_display_tracker,
                                            tracker
                                            )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_video(conf, model, video_path):
    """
    Plays a video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        video_path: Path to the video file.

    Returns:
        None

    Raises:
        None
    """
    vid_cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            _display_detected_frames(conf, model, st_frame, image)
        else:
            vid_cap.release()
            break


def video_clsifiction(conf, model):
    """
    Plays a stored video file or an uploaded video. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    uploaded_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        video_path = f"uploaded_video.{uploaded_file.name.split('.')[-1]}"
        with open(video_path, 'wb') as video_file:
            video_file.write(uploaded_file.read())
    else:
        st.sidebar.info("Please upload a video file.")
        return

    is_display_tracker, tracker = display_tracker_options()

    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    # Create a folder to save snapshots
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    snapshots_folder = f"{settings.SNAPSHOTS}/{video_name}"
    os.makedirs(snapshots_folder, exist_ok=True)

    detection_results_table_name = f'results_{uploaded_file.name}'
    detection_results_table = db.table(detection_results_table_name)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()

            counter = 0

            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    # Resize the image to a standard size
                    image_resized = cv2.resize(image, (720, int(720*(9/16))))

                    # Display the detected objects on the video frame
                    res = model.predict(image_resized, conf=conf)
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)

                    # Check if any object is detected
                    if len(res[0].boxes) > 0:
                        # Saveing  snapshot
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        snapshot_path = os.path.join(snapshots_folder, f"snapshot_{timestamp}.png")
                        cv2.imwrite(snapshot_path, cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

                        if counter <= 5:
                            #sending emergency email
                            try:
                                sos.send_emergency_email(video_name, timestamp, snapshot_path)
                            except Exception as e:
                                pass 

                        # Saveing snapshot path and timestamp to TinyDB
                        detection_results_table.insert({
                            'video_name': video_name,
                            'timestamp': timestamp,
                            'snapshot_path': snapshot_path,
                            'class_and_vacales': res[0].names[0],
                        })

                        counter += 1

                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error processing video: " + str(e))