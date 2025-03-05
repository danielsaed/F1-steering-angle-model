import streamlit as st
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from utils.video_processor import VideoProcessor,convert_video_to_10fps
from utils.model_handler import ModelHandler
from utils.ui_components import (
    create_header,
    create_upload_section,
    create_frame_selector,
    display_results,
    create_line_chart
)
from utils.video_processor import profiler

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_upload_section():
    """Create the video upload section"""
    st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov'],
        help="Upload onboard camera footage for analysis"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file

load_css()
cont = 0

col1, col2,col3 = st.columns([1,3,1])

with col2:
    st.title("F1 Steering Angle Prediction Model")

    '''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/danielsaed) 
    [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/daniel-saed/f1-steering-angle)
    '''
    tabs = st.tabs(["Prediction", "About the model"])
    # Initialize session state
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = ModelHandler()
    if 'fps_target' not in st.session_state:
        st.session_state.fps_target = 10  # Default FPS target
    if 'start_frame' not in st.session_state:
        st.session_state.start_frame = 0  # Default FPS target
    if 'end_frame' not in st.session_state:
        st.session_state.end_frame = 1  # Default FPS target
        
    with tabs[0]:  # Steering Angle Detection tab
        st.warning("Downloading or recording copyrighted content without permission may result in account restrictions or legal consequences. Users are responsible for ensuring compliance with applicable policies and laws within F1TV's Terms of Service.")
        coll1, coll2,coll3 = st.columns([12,1,8])
        with coll1:
            st.markdown("#### Step 1: Upload F1 Onboard Video ⬆️")
            st.markdown("")
            st.markdown("Recomendations to better model performance:")  
            st.markdown("- Maximun 1 lap video due to 200mb restriction, record all onboard video screen")
            st.markdown("- 1280 x 720 resolution or more, 30 FPS, 16/9 aspect ratio")
            uploaded_file = create_upload_section()

        

        with coll3:
            st.markdown("<span style='margin-right: 18px;'><strong>Onboard video example:</strong></span>", unsafe_allow_html=True)
            st.markdown("( For testing, if needed )", unsafe_allow_html=True)

            
            VIDEO_URL = str(Path("assets") / "demo_video.mp4")
            st.video(VIDEO_URL)

        st.markdown("")
        st.markdown("")
        st.markdown("")

        if uploaded_file:
            # Load video
            if st.session_state.video_processor.load_video(uploaded_file):
                total_frames = st.session_state.video_processor.total_frames
                original_fps = st.session_state.video_processor.fps
                
                # FPS selection dropdown - after video is loaded
                st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)

                # Create options at intervals of 5 FPS, plus 1 FPS option for fine-grained control
                # Cap at the original FPS of the video
                fps_options = [1]  # Start with 1 FPS
                for fps in range(5, original_fps + 1, 5):
                    fps_options.append(fps)
                    
                # Remove duplicates and sort
                fps_options = sorted(list(set(fps_options)))

                # Find the index of 10 FPS or the closest available option
                default_index = 0
                if 10 in fps_options:
                    default_index = fps_options.index(10)
                else:
                    closest_fps = min(fps_options, key=lambda x: abs(x - 10))
                    default_index = fps_options.index(closest_fps)

                st.markdown("#### Step 2: Select Start And End Frames ✂️")
                # Frame selection
                actual_start_frame = st.session_state.start_frame
                actual_end_frame = st.session_state.end_frame
                start_frame, end_frame = create_frame_selector(total_frames)

                st.session_state.start_frame = start_frame
                st.session_state.end_frame = end_frame

                if start_frame != actual_start_frame or end_frame != actual_end_frame:
                    st.session_state.btn = False

                
                # Preview frames side by side
                st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='section-title'>
                        Frame Preview
                    </div>
                """, unsafe_allow_html=True)
                
                preview_cols = st.columns(2)
                
                # Start frame preview
                with preview_cols[0]:
                    start_preview = st.session_state.video_processor.get_frame(start_frame)
                    if start_preview is not None:
                        st.image(start_preview, caption=f"Start Frame: {start_frame}")
                
                # End frame preview
                with preview_cols[1]:
                    end_preview = st.session_state.video_processor.get_frame(end_frame)
                    if end_preview is not None:
                        st.image(end_preview, caption=f"End Frame: {end_frame}")

                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display the current range information
                selected_frames = end_frame - start_frame + 1
                selected_duration = selected_frames / original_fps
                estimated_selected_frames = int(selected_duration * st.session_state.fps_target)

                
                                # Create a dropdown for FPS selection
                actual_fps = st.session_state.fps_target

                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("#### Step 3: Select FPS 👈")
                st.session_state.fps_target = st.selectbox(
                    "Select frames per second to process",
                    options=fps_options,
                    index=default_index,
                    format_func=lambda x: f"{x} FPS",
                    help="Choose how many frames per second to extract for processing"
                )

                if st.session_state.fps_target != actual_fps:
                    st.session_state.btn = False

                
                st.info(f"Selected range: {start_frame} to {end_frame} ({int(selected_duration*st.session_state.fps_target)} frames, {selected_duration:.2f} seconds). " 
                      f"At {st.session_state.fps_target} FPS")

                # Process button
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("#### Step 4: Execute Model 🚀")
                if st.button("Process Video Segment") or st.session_state.get('btn', True):

                    if not(st.session_state.get('btn', True)):
                        # Reset profiler before processing
                        profiler.reset()
                        
                        with st.spinner("Processing frames..."):
                            if int(selected_duration*st.session_state.fps_target) > 500:
                                st.warning("⚠️ Large video segment selected, it could take some minutes to process.")
            

                            # Extract and process frames
                            frames,crude_frames = st.session_state.video_processor.extract_frames(
                                start_frame, end_frame, fps_target=st.session_state.fps_target
                            )
                            results = st.session_state.model_handler.process_frames(
                                frames, "F1 Steering Angle Detection"
                            )
                            #st.session_state.processed_frames = crude_frames
                            #st.session_state.processed_frames1 = frames
                            # Convert results to DataFrame and display
                            df = st.session_state.model_handler.export_results(results)
                            st.session_state.df = df
                            # Create steering angle chart using Plotly
                    df = st.session_state.df
                    #crude_frames = st.session_state.processed_frames
                    #frames = st.session_state.processed_frames1
                    
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    
                    st.markdown("# Results")

                    display_results(df)

                    st.markdown("")
                    st.subheader("Steering Line Chart 📈")
                    # Create a Plotly figure

                    create_line_chart(df)

                    # Add steering angle statistics using Streamlit's built-in components
                    st.subheader("Steering Statistics 📊")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Mean Angle", f"{df['steering_angle'].mean():.2f}°")

                    with col2:
                        st.metric("Max Right Turn", f"{df['steering_angle'].max():.2f}°")

                    with col3:
                        st.metric("Max Left Turn", f"{df['steering_angle'].min():.2f}°")

                    with col4:
                        # Calculate average rate of change of steering angle
                        angle_changes = abs(df['steering_angle'].diff().dropna())
                        st.metric("Avg. Change Rate", f"{angle_changes.mean():.2f}°/frame")
                    st.session_state.btn = True

                    # Display profiler statistics in a new section
                    st.markdown("")

                    #st.markdown("# Performance Analysis")
                    
                    #stats, total_time = profiler.get_stats_dict()
                    

                            # Display the tabular results
                    

                    # Create Animation section with fixed parameters
                    st.write("")

                    st.markdown("</div>", unsafe_allow_html=True)


        else:
            st.session_state.btn = False
    
        
    with tabs[1]:  # Driver Behavior tab
        #st.markdown("## About the model")
        st.warning("This app does not endorse or encourage any actions that violate F1TV's Terms of Service. Downloading or recording copyrighted content without permission may result in account restrictions or legal consequences. Users are responsible for ensuring compliance with applicable policies and laws.")
        st.info("This model is for research/educational purposes only, its not related to F1 or any organization.")

        st.markdown("""
        ### The Model 
        
        In my opinion, steering inputs are a one of the key fundamental insights into driving behavior, performance and style. However,  like the full brake input, there is no public API or tool to access steering angle data. The only available source is onboard camera footage, which comes with its own limitations, such as camera position, shadows, weather conditions, and lighting. Despite these challenges, I think we can work around them to extract valuable insights.
                    
        - The **F1 Steering Angle Prediction Model** is a Convolutional Neural Network (CNN) based on EfficientNet-B0 with a regression head for angles from -180° to 180° to predict steering angles from a F1 onboard camera footage ( only for current gen F1 cars ), trained with over 1500 images, check **Technical Details** for image preprocesing.
        
        - Currentlly the model is able to predict steering angles with a decent accuracy, but it's still in development and will be improved in future versions, I recommend to use for analyse patterns, trends and other insights related, but not for precision steering angle analysis, for now.
        #####
        ### How It Works

        
        1. **Video Processing**: From the onboard camera video, frames are extracted at your selected FPS rate (Check video example on Prediction tab for reference)
        2. **Image Preprocessing**:
        - Cropping the image to focus on the track area
        - Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance visibility
        - Edge detection to highlight track boundaries
        3. **Neural Network Prediction**: A CNN model processes the edge image to predict the steering angle
        4. **Postprocessing**: apply local a trend-based outlier correction algorithm to detect and correct outliers
        4. **Results Visualization**: Angles are displayed as a line chart with statistical analysis also a excel with the frame and the steering angle.""")

        
        
                ### Model Architecture
        st.markdown("""
        #####
        ### Architecture
        Key features:
        
        - Input: 224x224px grayscale edge-detected images
        - Model: CNN with EfficientNet-B0 backbone and regression head
        - Output: Steering angle prediction between -180° and +180° with a local trend-based outlier correction algorithm
        - Training data: Over 1500+ Manually annotated F1 onboard footage      
        """)

        '''
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/danielsaed) 
        [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/daniel-saed/f1-steering-angle)
        '''

        st.markdown("""
        #####
        ### Performance Analysis
        
        After very long time and effort, the model has achieved the following performance metrics:
        
        - From 0 to +-90° = 6° of ground truth, from +-90° to +-180° = 13° of ground truth (this will improve in future versions, more images are needed to improve the model)
                    
        **Limitations**: Performance may decrease in:
        - Low visibility conditions (rain, extreme shadows)
        - Not well recorded videos (low resolution, high compression)
        - Change of onboard camera position (different angle, height)
        
        #####
        ### Image Technical Details
        
        The preprocessing pipeline is critical for good model performance:
        
        1. **Grayscale Conversion**: Reduces input size and complexity
        2. **Cropping**: Focuses on the track area for better predictions
        3. **Adaptive CLAHE**: Dynamically adjusts contrast to maximize track features visibility
        4. **Binary Thresholding**: Converts the image to binary for edge detection
        2. **Edge Detection**: Uses adaptive Canny edge detection targeting ~6% edge pixels per image
        3. **Model Format**: ONNX format for cross-platform compatibility and faster inference
        4. **Batch Processing**: Inference is done in batches for improved performance
        
        """)

        st.markdown("""
        #####
        ### Image Preprocessing Example""")
        col1, col2, col3 = st.columns([20,11.5,11.5])
        
        
        with col1:
            st.image("img/piastri-azerbaiyan_raw.jpg", caption="1. Original Frame")

        
        with col2:
            # Mostrar ejemplos de preprocesamiento - necesitas agregar estas imágenes a tu carpeta img/
            
            st.image("img/piastri-azerbaiyan_clahe.jpg", caption="2. After CLAHE Enhancement")
            
        with col3:
            st.image("img/piastri-azerbaiyan_edge.jpg", caption="3. Edge Detection (Model Input)")
        
        
        