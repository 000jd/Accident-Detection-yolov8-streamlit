from pathlib import Path
import pandas as pd
import json
import PIL
import streamlit as st
import utils.settings as settings_module
import utils.helper as helper_module

class AccidentDetectionApp:
    def __init__(self):
        # Set Streamlit page configuration
        st.set_page_config(
            page_title="Accident Detection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize settings and helper modules
        self.settings = settings_module.AccidentDetectionSettings()
        self.helper = helper_module.AccidentDetectionHelper()

        # Initialize model, confidence, source_radio, and source_img
        self.model = None
        self.confidence = None
        self.source_radio = None
        self.source_img = None

    def load_model(self, model_path):
        """
        Load the YOLO object detection model from the specified model_path.

        Parameters:
            model_path (str): The path to the YOLO model file.

        Returns:
            None
        """
        try:
            # Get the model name from the path
            model_name = Path(model_path).stem
            self.model = self.helper.load_model(model_name, model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

    def show_detection_page(self):
        """
        Display the main content for detection based on user-selected options.

        Returns:
            None
        """
        st.title("Accident Detection")

        # Sidebar
        st.sidebar.header("Navigation")

        # Navigation options
        page = st.sidebar.radio("Go to", ["Detection", "Results"])

        # Main content
        if page == "Detection":

            # Add dropdown for selecting the model
            selected_model_name = st.sidebar.selectbox("Select Model", [model[0] for model in self.settings.available_models])

            # Get the path of the selected model
            selected_model_path = next((model[1] for model in self.settings.available_models if model[0] == selected_model_name), None)

            if selected_model_path is None:
                st.error("No model selected.")
                return

            # Load the selected model
            self.load_model(selected_model_path)
            self.source_radio = st.sidebar.radio(
                    "Select Source", self.settings.SOURCES_LIST)
            
            # Sidebar options
            self.confidence = float(st.sidebar.slider(
                "Select Model Confidence", 25, 100, 40)) / 100

            # Main content for Detection
            if self.source_radio == self.settings.IMAGE:
                self.source_img = st.sidebar.file_uploader(
                    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

                col1, col2 = st.columns(2)

                with col1:
                    try:
                        if self.source_img is None:
                            default_image_path = str(self.settings.DEFAULT_IMAGE)
                            default_image = PIL.Image.open(default_image_path)
                            st.image(default_image_path, caption="Default Image",
                                    use_column_width=True)
                        else:
                            uploaded_image = PIL.Image.open(self.source_img)
                            st.image(self.source_img, caption="Uploaded Image",
                                    use_column_width=True)
                    except Exception as ex:
                        st.error("Error occurred while opening the image.")
                        st.error(ex)

                with col2:
                    if self.source_img is None:
                        default_detected_image_path = str(self.settings.DEFAULT_DETECT_IMAGE)
                        default_detected_image = PIL.Image.open(
                            default_detected_image_path)
                        st.image(default_detected_image_path, caption='Detected Image',
                                use_column_width=True)
                    else:
                        if st.sidebar.button('Detect Objects'):
                            res = self.model.predict(uploaded_image,
                                                    conf=self.confidence
                                                    )
                            boxes = res[0].boxes
                            res_plotted = res[0].plot()[:, :, ::-1]
                            st.image(res_plotted, caption='Detected Image',
                                    use_column_width=True)

                            try:
                                with st.expander(f"Detection {len(boxes)}"):
                                    for box in boxes:
                                        st.write(box.data)
                            except Exception as ex:
                                st.write("No image is uploaded yet!")

            elif self.source_radio == self.settings.VIDEO:
                self.helper.video_classification(self.confidence, self.model)

            elif self.source_radio == self.settings.DRONE:
                self.helper.cctv_camera_classification(self.confidence, self.model)

            else:
                st.error("Please select a valid source type!")
        elif page == "Results":
    
            st.sidebar.info("This is the Results page. Individual tables will be displayed for each video.")
            # Load detection results from the JSON file
            save_results_path = self.settings.DATABASE

            try:
                with open(save_results_path, 'r') as json_file:
                    detection_results = json.load(json_file)
            except Exception:
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
                        try:
                            video_names.append(result_details['video_name'])
                        except Exception as e:
                            video_names.append(result_details['video_id'])
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

    def run(self):
        """
        Run the accident detection app by loading the model and displaying the detection page.

        Returns:
            None
        """
        model_name, model_path = self.settings.available_models[0]  # Get the first available model name and path
        self.model = self.helper.load_model(model_name, Path(model_path))  # Load the model
        self.show_detection_page()

app = AccidentDetectionApp()
app.run()