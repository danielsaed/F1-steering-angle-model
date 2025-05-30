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
    #print("BASE_DIR", BASE_DIR)

if 'metrics_collection' not in st.session_state:
    from utils.helper import BASE_DIR,metrics_collection
    st.session_state.metrics_collection = metrics_collection
    #print("metrics_collection", metrics_collection)

BASE_DIR = st.session_state.BASE_DIR
metrics_collection = st.session_state.metrics_collection
path_load_css = Path(BASE_DIR) / "assets" / "style.css"
#print(path_load_css)

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
    keys_to_clear = ['video_processor', 'df', 'processed_frames', 'processed_frames1','end_dic', 'start_dic', 'start_preview', 'end_preview', 'start_preview1','end_frame_helper', 'start_frame_helper', 'start_frame', 'end_frame','driver_crop_type', 'driver_crop_type_2', 'start_frame_helper', 'end_frame_helper','postprocessing_mode']

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
    [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/daniel-saed/F1-steering-angle-dataset)

    '''
    tabs = st.tabs(["Use Model", "About"])
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
    if 'postprocessing_mode' not in st.session_state:
        st.session_state.postprocessing_mode = None



    with tabs[0]:  # Steering Angle Detection tab
        st.warning("Downloading or recording F1 onboards videos potentially violates F1/F1TV's terms of service.")
        coll1, coll2,coll3 = st.columns([12,1,8])
        with coll1:
            st.markdown("#### Step 1: Upload F1 Onboard Video ⬆️")
            st.markdown("")
            st.markdown("Recomendations:")  
            
            st.markdown("- Check historical DB before, the lap may already be processed.")
            st.markdown("- To record disable hardware aceleration on chrome.")
            st.markdown("- Onboards with no steering wheel visibility, like Leclerc's 2025, may not work well.")


            uploaded_file = create_upload_section()

        with coll3:
            st.markdown("<span style='margin-right: 18px;'><strong>Onboard example:</strong></span>", unsafe_allow_html=True)
            st.markdown("- 1080p,720p,480p resolutions, 10 to 30 FPS.")
            st.markdown("- Full onboard (mandatory).")
            #st.markdown("( For testing, if needed )", unsafe_allow_html=True)

            VIDEO_URL = Path(BASE_DIR) / "assets" / "demo_video.mp4"
            st.video(VIDEO_URL)


        if uploaded_file:
            with st.spinner("Loading..."):
                print("Video uploaded")
                
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.markdown("")
            # Load video
                if st.session_state.video_processor.load_video(uploaded_file):
                    

                    if st.session_state.end_dic is None:
                        st.session_state.end_dic = st.session_state.video_processor.frames_list_end
                        #print("End dic loaded:")

                    if st.session_state.start_dic is None:
                        st.session_state.start_dic = st.session_state.video_processor.frames_list_start
                        #print("Start dic loaded:")
                    
                    
                    total_frames = st.session_state.video_processor.total_frames
                    original_fps = st.session_state.video_processor.fps
                    #print("Original FPS:", original_fps)
                    #print("Total frames:", total_frames)


                    
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
                            value=st.session_state.start_frame_helper,
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
                            value=st.session_state.end_frame_helper,
                            step=1,
                            help="Select the start frame for processing",
                            key="end_frame_slider"
                        )
                        

                    
                    # Botones de control en la parte superior
                    btn_cols = st.columns([1, 1, 5, 1, 1, 5])
                    
                    with btn_cols[0]:
                        if st.button("-1",key="start_minus_1",use_container_width=True):
                            st.session_state.start_frame_helper = max(start_frame_min, st.session_state.start_frame_helper - 1)
                            st.rerun()  # Rerun to update the UI with the new value   
                    with btn_cols[1]:
                        if st.button("+1",key="start_plus_1",use_container_width=True):
                            st.session_state.start_frame_helper = min(start_frame_max, st.session_state.start_frame_helper + 1)
                            st.rerun()  # Rerun to update the UI with the new value       
                    with btn_cols[3]:
                        if st.button("-1", key="end_minus_1",
                                help="Decrease end frame by 1",
                                use_container_width=True):
                            st.session_state.end_frame_helper = max(end_frame_min, st.session_state.end_frame_helper - 1)
                            st.rerun()  # Rerun to update the UI with the new value  
                    with btn_cols[4]:
                        if st.button("+1", key="end_plus_1",
                                help="Increase end frame by 1", 
                                use_container_width=True):
                            st.session_state.end_frame_helper = min(end_frame_max, st.session_state.end_frame_helper + 1)
                            st.rerun()


                    #print("Start frame helper:", st.session_state.end_frame_helper)
                    #print("Start frame helper:", st.session_state.end_frame)

                    
                    
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
                                #print("Getting start frame preview for frame:", st.session_state.start_frame_helper)
                                st.session_state.start_preview = st.session_state.start_dic[st.session_state.start_frame_helper]
                                # Actualizar también el valor de referencia en session_state
                                st.session_state.start_frame = st.session_state.start_frame_helper
                            except Exception as e:
                                print("Error getting start frame preview:", e)
                                pass
                        
                        if st.session_state.start_preview is not None:
                            #print("Displaying start frame preview for frame:", st.session_state.start_frame_helper)
                            st.image(st.session_state.start_preview, caption=f"Start Frame: {st.session_state.start_frame_helper}", use_container_width=True)
                    
                    # End frame preview
                    with preview_cols[1]:
                        
                        # Aplicar la misma lógica para el end frame
                        if (st.session_state.end_preview is None or 
                            st.session_state.end_frame_helper != st.session_state.end_frame):
                            try:
                                #print("Getting end frame preview for frame:", st.session_state.end_frame_helper)
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


                    driver_crop_type = st.session_state.driver_crop_type

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
                            #st.session_state.driver_crop_type = driver_crop_type


                        
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
                            st.image(Path(BASE_DIR) / "img" / "example.png", caption=f"GOAL Frame:",use_container_width=True)

                        
                        

                        # Process button
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")

                        st.markdown("#### Step 5: (Opcional) Postprocessing Settings")
                        st.markdown("- First try default mode, is the best for 90% of the cases")
                        postprocessing_mode = st.session_state.postprocessing_mode

                        #agregar opciones en radio para elegir el tipo de procesamiento

                        st.session_state.postprocessing_mode = st.radio(
                            "Select Postprocessing Mode",
                            options=["Default","Low ilumination"],
                            index=0,
                            help="Choose the postprocessing mode for the model",
                            horizontal=False
                        )
                        if postprocessing_mode != st.session_state.postprocessing_mode:
                            
                            st.session_state.btn = False
                            #print("ininini")
                            
                            #st.rerun()  # Rerun to update the UI with the new value


                        # Process button
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")
                        st.markdown("")

                        st.markdown("#### Step 4: Execute Model 🚀")
                        if st.button("Process Video Segment") or st.session_state.get('btn', True):
                            
                            if not(st.session_state.get('btn', True)):
                                # Reset profiler before processing
                                profiler.reset()
                                print("Processing video...")
                                #st.rerun()  # Rerun to update the UI with the new value
                                
                                with st.spinner("Processing frames..."):
                                    if int(selected_duration*st.session_state.fps_target) > 500:
                                        st.warning("⚠️ Large video segment selected, it could take some minutes to process.")
                    

                                    # Extract and process frames
                                    st.session_state.video_processor.mode = st.session_state.postprocessing_mode
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
                            st.markdown("")
                            #st.markdown("#### Download Results 📥")
                            #if st.button("Download Results (CSV)"):
                            #    st.session_state.btn = True
                            #    df.to_csv(str(st.session_state.driver_crop_type)+"_Steering_data_results.csv", index=False)
                            #    st.info("Results downloaded successfully! you can find the file in your current directory.")

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
            print("No video uploaded yet.")
            st.session_state.btn = False
            try:
                st.session_state.video_processor.clean_up()  # Clear cache if no video is uploaded
                clear_session_state()
                print("Session state cleared")
            except:
                print("Error clearing session state")
    
        
    with tabs[1]:  # Driver Behavior tab
        
        st.info("For research/educational purposes only, its not related to F1 or any organization.")


        
        
        
        st.markdown("""
        #####
        ## The Model 
                         
        - The **F1 Steering Angle Prediction Model** uses a CNN based on EfficientNet-B0 to predict steering angles from a F1 onboard camera footage, trained with over 25,000 images (7000 manual labaled augmented to 25000) and YOLOv8-seg nano for helmets segmentation, allowing the model to be more robust by erasing helmet designs.
        
        - Currentlly the model is able to predict steering angles from 180° to -180° with a 3°-5° of error on ideal contitions.
                    
        - EfficientNet-B0 and YOLOv8-seg nano are exported to ONNX format, and images are resized to 224x224 allowing it to run on low-end devices.
                    
        
                    
        #####
        ## How It Works
        
        ##### Video Processing: 
        - From the onboard camera video, the frames selected are extracted at the FPS rate.
                    
        ##### Image Preprocessing:
        - The frames are cropeed based on selected crop type to focus on the steering wheel and driver area.
        - YOLOv8-seg nano is applied to the cropped images to segment the helmet, removing designs and logos.
        - Convert cropped images to grayscale and apply CLAHE to enhance visibility.
        - Apply adaptive Canny edge detection to extract edges, helped with preprocessing techniques like bilateralFilter and morphological transformations.
    
        ##### Prediction: 
        - The CNN model processes the edge image to predict the steering angle
                    
        ##### Postprocessing
        - apply local a trend-based outlier correction algorithm to detect and correct outliers
                    
        ##### Results Visualization
        - Angles are displayed as a line chart with statistical analysis also a csv file with the frame number, time and the steering angle.
        #####""")


        coll1, coll2, coll3,coll4,coll5 = st.columns([40,23,23,23,23])

        with coll1:
            st.image(Path(BASE_DIR) / "img" / "verstappen_china_2025.jpg", caption="1. Original Frame", use_container_width=True)
        with coll2:
            # Mostrar ejemplos de preprocesamiento - necesitas agregar estas imágenes a tu carpeta img/
            
            st.image(Path(BASE_DIR) / "img" / "verstappen_china_2025_cropped.jpg", caption="2. Crop image",use_container_width=True)

            
        with coll3:
            st.image(Path(BASE_DIR) / "img" / "verstappen_china_2025_nohelmet.jpg", caption="3. Segment Helmet with YOLO",use_container_width=True)
        
        with coll4:
            st.image(Path(BASE_DIR) / "img" / "verstappen_china_2025_clahe.jpg", caption="4. Apply clahe",use_container_width=True)
        
        with coll5:
            st.image(Path(BASE_DIR) / "img" / "verstappen_china_2025_tresh.jpg", caption="5. Edge detection",use_container_width=True)

                    
        st.markdown("""
        ####
        ## Limitations 
        - Low visibility conditions (rain, extreme shadows, extreme light).
        - Not well recorded videos.
        - Change of onboard camera position (different angle, height, shakiness).
                    """)

        

        
        
        
        
        
