import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from utils.video_processor import VideoProcessor
from utils.model_handler import ModelHandler
from utils.ui_components import (
    create_header,
    create_upload_section,
    create_frame_selector,
    display_results,
    create_line_chart
)
from utils.video_processor import profiler


if 'BASE_DIR' not in st.session_state:
    from utils.helper import BASE_DIR,metrics_collection
    st.session_state.BASE_DIR = BASE_DIR
    print("BASE_DIR", BASE_DIR)

if 'metrics_collection' not in st.session_state:
    from utils.helper import BASE_DIR,metrics_collection
    st.session_state.metrics_collection = metrics_collection
    print("metrics_collection", metrics_collection)

BASE_DIR = st.session_state.BASE_DIR
metrics_collection = st.session_state.metrics_collection
path_load_css = Path(BASE_DIR) / "assets" / "style.css"
print(path_load_css)

def load_css():
    with open(Path(BASE_DIR) / "assets" / "style.css") as f:
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


def clear_session_state():
    """Clear unnecessary session state variables to free memory."""
    keys_to_clear = ['video_processor', 'df', 'processed_frames', 'processed_frames1','end_dic', 'start_dic', 'start_preview', 'end_preview', 'start_preview1','end_frame_helper', 'start_frame_helper', 'start_frame', 'end_frame','driver_crop_type', 'driver_crop_type_2']

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


load_css()
cont = 0



col1, col2,col3 = st.columns([1,3,1])

with col2:
    st.title("F1 Steering Angle Model")

    '''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/danielsaed/F1-steering-angle-predictor) 
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
    if 'driver_crop_type' not in st.session_state:
        st.session_state.driver_crop_type = None  # Default FPS target
    if 'driver_crop_type_2' not in st.session_state:
            st.session_state.driver_crop_type_2 = None  # Default FPS target
    if 'start_frame' not in st.session_state:
        st.session_state.start_frame = 0  # Default FPS target
    if 'end_frame' not in st.session_state:
        st.session_state.end_frame = -1  # Default FPS target
    if 'start_preview' not in st.session_state:
        st.session_state.start_preview = None
    if 'end_preview' not in st.session_state:
        st.session_state.end_preview = None
    if 'start_preview1' not in st.session_state:
        st.session_state.start_preview1 = None
    if 'start_frame_helper' not in st.session_state:
        st.session_state.start_frame_helper = 0
    if 'end_frame_helper' not in st.session_state:
        st.session_state.end_frame_helper = -1
    if 'end_dic' not in st.session_state:
        st.session_state.end_dic = None
    if 'start_dic' not in st.session_state:
        st.session_state.start_dic = None



    with tabs[0]:  # Steering Angle Detection tab
        st.warning("Downloading or recording F1 onboards videos potentially violates F1/F1TV's terms of service.")
        coll1, coll2,coll3 = st.columns([12,1,8])
        with coll1:
            st.markdown("#### Step 1: Upload F1 Onboard Video ⬆️")
            st.markdown("")
            st.markdown("Recomendations:")  
            st.markdown("- 1 lap videos, full onboard screen")
            st.markdown("- 1080p,720p,480p resolutions, 10 to 30 FPS, 16/9 aspect ratio, consider 200mb restriction")

            uploaded_file = create_upload_section()

        with coll3:
            st.markdown("<span style='margin-right: 18px;'><strong>Onboard video example:</strong></span>", unsafe_allow_html=True)
            #st.markdown("( For testing, if needed )", unsafe_allow_html=True)

            VIDEO_URL = Path(BASE_DIR) / "assets" / "demo_video.mp4"
            st.video(VIDEO_URL)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

        if uploaded_file:
            with st.spinner("Loading video..."):
            # Load video
                if st.session_state.video_processor.load_video(uploaded_file):
                    

                    if st.session_state.end_dic is None:
                        st.session_state.end_dic = st.session_state.video_processor.frames_list_end
                        print("End dic loaded:")

                    if st.session_state.start_dic is None:
                        st.session_state.start_dic = st.session_state.video_processor.frames_list_start
                        print("Start dic loaded:")
                    
                    
                    total_frames = st.session_state.video_processor.total_frames
                    original_fps = st.session_state.video_processor.fps
                    print("Original FPS:", original_fps)
                    print("Total frames:", total_frames)


                    
                    # FPS selection dropdown - after video is loaded
                    st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)


                    st.markdown("#### Step 2: Select Start And End Frames ✂️")


                    start_frame_min = 0
                    start_frame_max = int(total_frames * 0.1)  # 10% del total
                    end_frame_min = int(total_frames * 0.9)    # 90% del total
                    end_frame_max = total_frames - 1
                    if st.session_state.end_frame_helper == -1:
                        st.session_state.end_frame_helper = end_frame_max
                    
                    # Actualizar los valores de session_state basándose en el slider
                    st.markdown("- Match start & finish line")
                    # CSS personalizado para los botones
                    preview_cols1 = st.columns(2)

                    with preview_cols1[0]:
                        st.markdown("##### Start Frame")
                        #slicer
                        st.session_state.start_frame_helper = st.slider(
                            "Select Start Frame",
                            min_value=st.session_state.video_processor.start_frame_min,
                            max_value=st.session_state.video_processor.start_frame_max,
                            value=0,
                            step=1,
                            help="Select the start frame for processing",
                            key="start_frame_slider"
                        )
                    with preview_cols1[1]:
                        st.markdown("##### End Frame")
                        st.session_state.end_frame_helper = st.slider(
                            "Select Start Frame",
                            min_value=st.session_state.video_processor.end_frame_min,
                            max_value=st.session_state.video_processor.end_frame_max,
                            value=st.session_state.video_processor.end_frame_max,
                            step=1,
                            help="Select the start frame for processing",
                            key="end_frame_slider"
                        )
                        

                    
                    # Botones de control en la parte superior
                    btn_cols = st.columns([1, 1, 5, 1, 1, 5])
                    
                    with btn_cols[0]:
                        if st.button("-1",key="start_minus_1",use_container_width=True):
                            st.session_state.start_frame_helper = max(start_frame_min, st.session_state.start_frame_helper - 1)   
                    with btn_cols[1]:
                        if st.button("+1",key="start_plus_1",use_container_width=True):
                            st.session_state.start_frame_helper = min(start_frame_max, st.session_state.start_frame_helper + 1)       
                    with btn_cols[3]:
                        if st.button("-1", key="end_minus_1",
                                help="Decrease end frame by 1",
                                use_container_width=True):
                            st.session_state.end_frame_helper = max(end_frame_min, st.session_state.end_frame_helper - 1)  
                    with btn_cols[4]:
                        if st.button("+1", key="end_plus_1",
                                help="Increase end frame by 1", 
                                use_container_width=True):
                            st.session_state.end_frame_helper = min(end_frame_max, st.session_state.end_frame_helper + 1)
                            #st.rerun()
                    
                    
                    # Añadir un poco de espacio entre botones y previsualizaciones
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Preview columns originales (mantener tu código existente)
                    preview_cols = st.columns(2)
                    
                    # Start frame preview
                    with preview_cols[0]:
                    
                        # Siempre verificar si necesitamos actualizar la previsualización
                        if (st.session_state.start_preview is None or 
                            st.session_state.start_frame_helper != st.session_state.start_frame):
                            try:
                                print("Getting start frame preview for frame:", st.session_state.start_frame_helper)
                                st.session_state.start_preview = st.session_state.start_dic[st.session_state.start_frame_helper]
                                # Actualizar también el valor de referencia en session_state
                                st.session_state.start_frame = st.session_state.start_frame_helper
                            except Exception as e:
                                print("Error getting start frame preview:", e)
                                pass
                        
                        if st.session_state.start_preview is not None:
                            print("Displaying start frame preview for frame:", st.session_state.start_frame_helper)
                            st.image(st.session_state.start_preview, caption=f"Start Frame: {st.session_state.start_frame_helper}", use_container_width=True)
                    
                    # End frame preview
                    with preview_cols[1]:
                        
                        # Aplicar la misma lógica para el end frame
                        if (st.session_state.end_preview is None or 
                            st.session_state.end_frame_helper != st.session_state.end_frame):
                            try:
                                print("Getting end frame preview for frame:", st.session_state.end_frame_helper)
                                st.session_state.end_preview = st.session_state.end_dic[st.session_state.end_frame_helper]
                                # Actualizar también el valor de referencia en session_state
                                st.session_state.end_frame = st.session_state.end_frame_helper
                            except Exception as e:
                                print("Error getting end frame preview:", e)
                                pass
                        if st.session_state.end_preview is not None:
                            st.image(st.session_state.end_preview, caption=f"End Frame: {st.session_state.end_frame_helper}", use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)
    # ...existing code...
                    
                    # Display the current range information
                    selected_frames = st.session_state.end_frame_helper - st.session_state.start_frame_helper + 1
                    selected_duration = selected_frames / original_fps
                    estimated_selected_frames = int(selected_duration * st.session_state.fps_target)

                    # Create a dropdown for FPS selection
                    actual_fps = st.session_state.fps_target
                    st.session_state.fps_target = original_fps

                    st.info(f"Selected range: {st.session_state.start_frame_helper} to {st.session_state.end_frame_helper} ({int(selected_duration*st.session_state.fps_target)} frames, {selected_duration:.2f} seconds). " 
                        f"At {st.session_state.fps_target} FPS")

                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")

                    
                    
                    lst_team_option = ('RedBull', 'Ferrari', 'Mclaren','Mercedes','Williams','Aston Martin','RB','Hass','Sauber', 'Alpine')

                    dic_masks = {
                        'RedBull': ('Verstappen 2025','Tsunoda 2025'),
                        'Ferrari': ('Hamilton 2025','Leclerc 2025'),
                        'Mclaren': ('Piastri 2025','Norris 2025'),
                        'Mercedes': ('Antonelli 2025','Russell 2025'),
                        'Williams': ('Albon 2025','Sainz 2025'),
                        'Alpine': ('Gasly 2025','Colapinto 2025'),
                        'RB': ('Hadjar 2025','Lawson 2025'),
                        'Hass': ('Bearman 2025','Ocon 2025'),
                        'Sauber': ('Hulk 2025','Bortoleto 2025'),
                        'Aston Martin': ('Alonso 2025','Stroll 2025')


                    }
                    
                    #('Verstappen 2025', 'Piastri 2025','Norris 2025','Leclerc 2025','Hamilton 2025','Russell 2025', 'Antonelli 2025', 'Tsunoda 2025')


                    driver_crop_type = st.session_state.driver_crop_type_2

                    st.markdown("#### Step 3: Select Crop type 👈")
                    st.markdown("- Steering wheel, helmet and hands shold be visible, aim for acrop type like the example image.")
                    st.markdown("- Some onboards change the camera position along the season, a different team/driver crop type can match the camera position desired.")
                    

                    lst_columns = st.columns(2)

                    with lst_columns[0]:
                            
                        st.session_state.driver_crop_type_2 = st.selectbox(
                            "Select team",
                            lst_team_option,
                            index=None,
                            format_func=lambda x: f"{x}",
                            help="Choose recort for processing"
                        )
                    with lst_columns[1]:
                        if st.session_state.driver_crop_type_2 != None:
                            st.session_state.driver_crop_type = st.selectbox(
                                "Select driver",
                                dic_masks[st.session_state.driver_crop_type_2],
                                index=0,
                                format_func=lambda x: f"{x}",
                                help="Choose recort for processing"
                            )
                    
                    if st.session_state.driver_crop_type != None:
                        # Update the video processor with the selected crop type
                        
                        print("Crop type updated to:", st.session_state.driver_crop_type)
                    
                        if st.session_state.driver_crop_type != driver_crop_type:
                                
                            st.session_state.btn = False

                        
                        preview_cols1 = st.columns(2)
                        with preview_cols1[0]:
                            st.markdown("##### Current Crop Type")
                            if st.session_state.start_preview1 is None or st.session_state.driver_crop_type != driver_crop_type:
                                st.session_state.start_preview1 = st.session_state.video_processor.get_frame_example(0)
                                st.session_state.video_processor.load_crop_variables(st.session_state.driver_crop_type)
                                
                                st.session_state.start_preview1 = st.session_state.video_processor.crop_frame_example(st.session_state.start_preview1)
                                
                            if st.session_state.start_preview1 is not None:
                                st.image(st.session_state.start_preview1, caption=f"Example",use_container_width=True)
                        
                        # End frame preview
                        with preview_cols1[1]:
                            st.markdown("##### Example frame")
                            #end_preview1 = st.session_state.video_processor.get_frame(end_frame)
                            #end_preview1 = cv2.imread("img\example.png")
                            
                            
                            st.image(Path(BASE_DIR) / "img" / "example.png", caption=f"GOAL Frame:",use_container_width=True)

                        
                        

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
                                        st.session_state.start_frame_helper, st.session_state.end_frame_helper, fps_target=st.session_state.fps_target
                                    )
                                    st.session_state.model_handler.fps = original_fps
                                    results = st.session_state.model_handler.process_frames(
                                        frames, "F1 Steering Angle Detection"
                                    )
                                    try:
                                        metrics_collection.update_one(
                                            {"action": "descargar_app"},
                                            {"$inc": {"count": 1}}
                                        )
                                    except:
                                        st.warning("MongoDB client not connected.")
                                    #st.session_state.processed_frames = crude_frames
                                    #st.session_state.processed_frames1 = frames
                                    # Convert results to DataFrame and display
                                    df = st.session_state.model_handler.export_results(results)
                                    st.session_state.df = df
                                    st.session_state.video_processor.clear_cache()  # Clear cache after processing
                                    # Clear unnecessary session state variables to free memory
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


        else:
            st.session_state.btn = False
            try:
                st.session_state.video_processor.clean_up()  # Clear cache if no video is uploaded
                clear_session_state()
                print("Session state cleared")
            except:
                print("Error clearing session state")
    
        
    with tabs[1]:  # Driver Behavior tab
        
        st.info("The model is for research/educational purposes only, its not related to F1 or any organization.")

        st.markdown("""
        ### The Model 
        
        Steering input is one of the key fundamental insights into driving behavior, performance and style. However, there is no straigfoward public source, tool or API to access steering angle data. The only available source is onboard camera footage, which comes with its own limitations, such as camera position, shadows, weather conditions, and lighting. Despite these challenges, I think we can work around them to extract valuable insights.
                    
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
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/danielsaed/F1-steering-angle-predictor) 
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
            st.image(Path(BASE_DIR) / "img" / "piastri-azerbaiyan_raw.jpg", caption="1. Original Frame")

        
        with col2:
            # Mostrar ejemplos de preprocesamiento - necesitas agregar estas imágenes a tu carpeta img/
            
            st.image(Path(BASE_DIR) / "img" / "piastri-azerbaiyan_clahe.jpg", caption="2. After CLAHE Enhancement")
            
        with col3:
            st.image(Path(BASE_DIR) / "img" / "piastri-azerbaiyan_edge.jpg", caption="3. Edge Detection (Model Input)")
        
        
