import json
from pathlib import Path
import PIL
import streamlit as st
from tinydb import TinyDB
import utils.settings as settings
import utils.helper as helper
import pandas as pd

st.set_page_config(
    page_title="Accident Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Accident Detection")

# Sidebar
st.sidebar.header("Navigation")

# Navigation options
page = st.sidebar.radio("Go to", ["Detection", "Results"])

# Main content
if page == "Detection":
    #st.title("Accident Detection")

    # Sidebar
    st.sidebar.header("Model Config")

    # Model Options
    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    # Selecting Detection
    model_path = Path(settings.DETECTION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image/Video Options")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    source_img = None
    # If image is selected

    # Main content for Detection
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    
                    # Save detection results to a text file
                    save_results_path = "detection.txt"
                    with open(save_results_path, "w") as results_file:
                        results_file.write("Detection Results:\n")
                        for box in enumerate(res):
                            results_file.write(f"{box}\n")
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        
    elif source_radio == settings.VIDEO:
        helper.video_clsifiction(confidence, model)

    elif source_radio == settings.RTSP:
        helper.play_rtsp_stream(confidence, model)

    else:
        st.error("Please select a valid source type!")

elif page == "Results":

    st.sidebar.info("This is the Results page. Individual tables will be displayed for each video.")
    
    # Load detection results from the JSON file
    save_results_path = settings.DATABASE

    try:
        with open(save_results_path, 'r') as json_file:
            detection_results = json.load(json_file)
    except FileNotFoundError:
        st.warning("No detection results found. Please run the detection first.")
        detection_results = {}

    if detection_results:
        st.write("## Detection Results")

        for video_name, results in detection_results.items():
            # Create lists to store data for the table
            video_names = []
            timestamps = []
            image_paths = []

            for result_id, result_details in results.items():
                # Append data to lists
                video_names.append(result_details['video_name'])
                timestamps.append(result_details['timestamp'])
                image_paths.append(result_details['snapshot_path'])

            # Create a DataFrame for the table
            table_data = {'Video Name': video_names, 'Timestamp': timestamps}
            df = pd.DataFrame(table_data)

            # Use st.columns to create a three-column layout
            col1, col2= st.columns(2)

            # Display the table in the first column
            with col1:
                st.write("### Table")
                st.table(df)

            # Display the images in the second and third columns
            with col2:
                st.write("### Images ")
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    for i in range(0, len(image_paths), 3):
                        st.image(image_paths[i], width=200)

                with sub_c2:
                    for i in range(1, len(image_paths), 3):
                        st.image(image_paths[i], width=200)
    else:
        st.warning("No detection results found. Please run the detection first.")



