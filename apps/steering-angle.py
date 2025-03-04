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
    display_results
)

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
    st.title("F1 Steering Angle Detection Model")

    '''
    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/danielsaed) 
    [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/daniel-saed/f1-steering-angle)
    '''
    tabs = st.tabs(["Prediction", "How to execute", "About this model"])
    # Initialize session state
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = ModelHandler()
    if 'fps_target' not in st.session_state:
        st.session_state.fps_target = 10  # Default FPS target
        
    with tabs[0]:  # Steering Angle Detection tab

        coll1, coll2,coll3 = st.columns([12,1,8])
        with coll1:
            st.markdown("#### Step 1: Upload F1 Onboard Video ⬆️")
            st.markdown("")
            st.markdown("Recomendations to better model performance:")  
            st.markdown("- Maximun 1 lap video, record all onboard video screen")
            st.markdown("- 1280 x 720 resolution or more, 30 FPS, 16/9 aspect ratio")
            uploaded_file = create_upload_section()

        

        with coll3:
            st.markdown("<span style='margin-right: 18px;'><strong>Video example:</strong></span>", unsafe_allow_html=True)
            st.markdown("(You can download it for testing, if needed)", unsafe_allow_html=True)

            
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
                start_frame, end_frame = create_frame_selector(total_frames)
                
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
                        with st.spinner("Processing frames..."):

                            # Extract and process frames
                            frames,crude_frames = st.session_state.video_processor.extract_frames(
                                start_frame, end_frame, fps_target=st.session_state.fps_target
                            )
                            results = st.session_state.model_handler.process_frames(
                                frames, "F1 Steering Angle Detection"
                            )
                            st.session_state.processed_frames = crude_frames
                            st.session_state.processed_frames1 = frames
                            # Convert results to DataFrame and display
                            df = st.session_state.model_handler.export_results(results)
                            st.session_state.df = df
                            # Create steering angle chart using Plotly
                    df = st.session_state.df
                    crude_frames = st.session_state.processed_frames
                    frames = st.session_state.processed_frames1
                    
                    st.markdown("")
                    st.markdown("")
                    st.markdown("")
                    
                    st.markdown("# Results")

                    display_results(df)
                    st.markdown("")
                    st.subheader("Steering Line Chart 📈")
                    # Create a Plotly figure
                    fig = go.Figure()

                    # Add the main steering angle line
                    fig.add_trace(go.Scatter(
                        x=df['frame_number'], 
                        y=df['steering_angle'],
                        mode='lines',
                        name='Steering Angle',
                        line=dict(color='blue', width=3),
                        hovertemplate='<b>Frame:</b> %{x}<br><b>Angle:</b> %{y:.2f}°<extra></extra>'
                    ))

                    # Add reference lines for straight, full right, and full left
                    fig.add_shape(type="line",
                        x0=df['frame_number'].min(), y0=0, x1=df['frame_number'].max(), y1=0,
                        line=dict(color="red", width=2, dash="solid"),
                        name="Straight (0°)"
                    )

                    fig.add_shape(type="line",
                        x0=df['frame_number'].min(), y0=90, x1=df['frame_number'].max(), y1=90,
                        line=dict(color="red", width=2, dash="dash"),
                        name="Full Right (90°)"
                    )

                    fig.add_shape(type="line",
                        x0=df['frame_number'].min(), y0=-90, x1=df['frame_number'].max(), y1=-90,
                        line=dict(color="red", width=2, dash="dash"),
                        name="Full Left (-90°)"
                    )

                    # Añadir etiquetas a las líneas de referencia
                    fig.add_annotation(x=df['frame_number'].min(), y=0,
                        text="Straight (0°)",
                        showarrow=True,
                        arrowhead=1,
                        ax=-40,
                        ay=-20
                    )

                    fig.add_annotation(x=df['frame_number'].min(), y=90,
                        text="Full Right (90°)",
                        showarrow=True,
                        arrowhead=1,
                        ax=-40,
                        ay=-20
                    )

                    fig.add_annotation(x=df['frame_number'].min(), y=-90,
                        text="Full Left (-90°)",
                        showarrow=True,
                        arrowhead=1,
                        ax=-40,
                        ay=20
                    )

                    # Configure layout
                    fig.update_layout(
                        title="Steering Angle Over Time",
                        xaxis_title="Frame Number",
                        yaxis_title="Steering Angle (degrees)",
                        yaxis=dict(range=[-180, 180]),
                        hovermode="x unified",
                        legend_title="Legend",
                        template="plotly_white",
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    # Add a light gray range for "straight enough" (-10° to 10°)
                    fig.add_shape(type="rect",
                        x0=df['frame_number'].min(), y0=-10,
                        x1=df['frame_number'].max(), y1=10,
                        fillcolor="lightgray",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                    )

                    # Display the plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

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


                            # Display the tabular results
                    

                    # Create Animation section with fixed parameters
                    st.write("")
                    '''st.subheader("Frames Preview")
                    st.write("Preview of the processed frames with steering angle overlay.")
                    st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)

                    if frames and len(frames) > 0 and crude_frames and len(crude_frames) > 0:
                        # Fixed parameters for the animation
                        gif_duration = int(st.session_state.fps_target*.25) if int(st.session_state.fps_target*.25) > 0 else 1  # FPS for slow motion
                        gif_quality = 30  # Quality
                        gif_size = 50     # Size percentage
                        s
                        # Create animation automatically
                        from PIL import Image
                        import io
                        import cv2
                        import numpy as np
                        
                        # Create two lists to store processed PIL images
                        pil_frames_processed = []  # For frames (processed)
                        pil_frames_original = []   # For crude_frames (original)
                        
                        with st.spinner("Preparing video previews with steering angle overlay..."):
                            # Process each frame - for both versions
                            
                            # Limitar a máximo 1000 frames
                            max_frames = 1000
                            total_frames = max(len(frames), len(crude_frames))
                            
                            # Si hay más de max_frames, tomar muestras distribuidas uniformemente
                            if total_frames > max_frames:
                                # Calcular el intervalo de muestreo para distribuir max_frames uniformemente
                                sample_interval = total_frames / max_frames
                                frame_indices = [int(i * sample_interval) for i in range(max_frames)]
                                st.info(f"Video has {total_frames} frames. Processing {max_frames} frames for the animation (1 frame every {sample_interval:.1f} frames).")
                            else:
                                frame_indices = range(total_frames)
                            
                            # Procesar solo los frames seleccionados
                            for idx in frame_indices:
                                # FIRST GIF - PROCESSED FRAMES
                                if idx < len(frames):
                                    frame = frames[idx]
                                    
                                    # Get the steering angle for this frame from the dataframe
                                    if idx < len(df):
                                        angle = df.iloc[idx]['steering_angle']
                                        
                                        # Create a copy of the frame to draw on
                                        annotated_frame = frame.copy()
                                        
                                        # Get dimensions for positioning
                                        height, width = annotated_frame.shape[:2]
                                        
                                        # Draw a semi-transparent black background for text
                                        cv2.rectangle(annotated_frame, (10, height-80), (360, height-10), (0, 0, 0), -1)
                                        cv2.rectangle(annotated_frame, (10, height-80), (360, height-10), (255, 255, 255), 2)
                                        
                                        # Add steering angle text
                                        cv2.putText(annotated_frame, 
                                                  f"Steering: {angle:.1f} deg", 
                                                  (20, height-40), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  1.0, (255, 255, 255), 2)
                                        
                                        # Draw a steering wheel indicator
                                        center_x, center_y = 180, height-40
                                        radius = 30
                                        
                                        # Calculate line endpoint for steering indicator
                                        angle_rad = np.radians(-angle)  # Negate for correct orientation
                                        end_x = int(center_x + radius * np.sin(angle_rad))
                                        end_y = int(center_y - radius * np.cos(angle_rad))
                                        
                                        # Convert the annotated frame to a PIL image
                                        img = Image.fromarray(annotated_frame)
                                    else:
                                        # If there's no angle data for this frame, just use the original
                                        img = Image.fromarray(frame)
                                    
                                    # Resize to 50% width
                                    width, height = img.size
                                    new_width = int(width * gif_size / 100)
                                    new_height = int(height * gif_size / 100)
                                    img = img.resize((new_width, new_height), Image.LANCZOS)
                                    
                                    pil_frames_processed.append(img)
                                
                                # SECOND GIF - ORIGINAL CRUDE FRAMES
                                if idx < len(crude_frames):
                                    frame = crude_frames[idx]
                                    
                                    # REDUCCIÓN AGRESIVA: Reducir drásticamente la calidad (al 15% del tamaño original)
                                    small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_NEAREST)
                                    
                                    # Get the steering angle for this frame from the dataframe
                                    if idx < len(df):
                                        angle = df.iloc[idx]['steering_angle']
                                        
                                        # Create a copy of the SMALL frame to draw on
                                        annotated_frame = small_frame.copy()
                                        
                                        # Get dimensions for positioning
                                        height, width = annotated_frame.shape[:2]
                                        
                                        
                                        # Convertir a imagen PIL
                                        img = Image.fromarray(annotated_frame)
                                    else:
                                        # Si no hay datos de ángulo para este frame, usar el original pequeño
                                        img = Image.fromarray(small_frame)
                                    
                                    # No usar LANCZOS aquí para ahorrar procesamiento
                                    width, height = img.size
                                    new_width = int(width * gif_size / 100)
                                    new_height = int(height * gif_size / 100)
                                    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)  # NEAREST es más rápido que LANCZOS
                                    
                                    pil_frames_original.append(img)
                            
                            # Create GIF buffers in memory
                            processed_gif_buffer = io.BytesIO()
                            original_gif_buffer = io.BytesIO()
                            
                            # Save the first GIF (processed frames)
                            if pil_frames_processed:
                                pil_frames_processed[0].save(
                                    processed_gif_buffer, 
                                    format='GIF',
                                    save_all=True,
                                    append_images=pil_frames_processed[1:],
                                    optimize=True,
                                    quality=gif_quality,
                                    duration=int(1000/gif_duration),  # Duration in milliseconds
                                    loop=0  # Loop forever
                                )
                                processed_gif_buffer.seek(0)
                            
                            # Save the second GIF (original frames)
                            if pil_frames_original:
                                pil_frames_original[0].save(
                                    original_gif_buffer, 
                                    format='GIF',
                                    save_all=True,
                                    append_images=pil_frames_original[1:],
                                    optimize=True,
                                    quality=gif_quality,
                                    duration=int(1000/gif_duration),  # Duration in milliseconds
                                    loop=0  # Loop forever
                                )
                                original_gif_buffer.seek(0)
                        
                        # Display GIFs in a vertical layout in the center column with 60% size
                        #st.markdown("### Video Preview")
                        st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)

                        # Aquí vamos a eliminar la visualización de los dos GIFs separados
                        # y saltar directamente a la creación del GIF combinado

                        # Create a central column for the combined GIF

                        with st.spinner("Creating synchronized comparison video..."):
                            # Create a list to store the combined frames
                            combined_frames = []
                            
                            # Limitar a máximo 1000 frames para el GIF combinado
                            max_gif_frames = 1000
                            total_combined_frames = min(len(pil_frames_processed), len(pil_frames_original))
                            
                            # Si hay más de max_gif_frames, tomar muestras distribuidas uniformemente
                            if total_combined_frames > max_gif_frames:
                                # Calcular el intervalo de muestreo
                                sample_interval = total_combined_frames / max_gif_frames
                                combined_indices = [int(i * sample_interval) for i in range(max_gif_frames)]
                                #st.info(f"Combining {max_gif_frames} frames for the animation (sampling 1 frame every {sample_interval:.1f} frames).")
                            else:
                                combined_indices = range(total_combined_frames)
                            
                            # Process each selected pair of frames
                            for i in combined_indices:
                                # Get both frames
                                proc_frame = pil_frames_processed[i]
                                orig_frame = pil_frames_original[i]
                                
                                # Make sure both frames have the same height
                                proc_width, proc_height = proc_frame.size
                                orig_width, orig_height = orig_frame.size
                                target_height = max(proc_height, orig_height)
                                
                                # Resize if necessary to match heights
                                if proc_height != target_height:
                                    new_width = int(proc_width * target_height / proc_height)
                                    proc_frame = proc_frame.resize((new_width, target_height), Image.Resampling.LANCZOS)
                                    
                                if orig_height != target_height:
                                    new_width = int(orig_width * target_height / orig_height)
                                    orig_frame = orig_frame.resize((new_width, target_height), Image.Resampling.LANCZOS)
                                
                                # Create a new blank image with enough space for both frames side by side
                                proc_width, proc_height = proc_frame.size
                                orig_width, orig_height = orig_frame.size
                                combined_width = proc_width + orig_width + 20  # 20px spacing between images
                                combined_height = target_height
                                
                                combined = Image.new('RGB', (combined_width, combined_height), (0, 0, 0))
                                
                                # Add processed frame on the left
                                combined.paste(proc_frame, (0, 0))
                                
                                # Add labels
                                draw = ImageDraw.Draw(combined)
                                # Add frame counter to make animation more visible
                                draw.rectangle([(10, 10), (150, 50)], fill=(0, 0, 0), outline=(255, 255, 255))
                                draw.text((20, 20), f"Frame: {i+1}/{len(pil_frames_processed)}", 
                                            fill=(255, 255, 0))
                                
                                # Add labels for each side
                                #draw.text((proc_width//2 - 70, target_height - 30), "Processed Frame", fill=(255, 255, 255))
                                draw.text((proc_width + 20 + orig_width//2 - 70, target_height - 30), "Original Frame", fill=(255, 255, 255))
                                
                                # Add a vertical separator line
                                for y in range(combined_height):
                                    if y % 2 == 0:  # Dashed line
                                        draw.point((proc_width + 10, y), fill=(255, 255, 255))
                                
                                # Add original frame on the right
                                combined.paste(orig_frame, (proc_width + 20, 0))
                                
                                # Add to the combined frames list
                                combined_frames.append(combined)
                            
                            # Create a GIF from the combined frames
                            combined_gif_buffer = io.BytesIO()
                            
                            if combined_frames:
                                # Save as GIF with optimized parameters for animation
                                combined_frames[0].save(
                                    combined_gif_buffer, 
                                    format='GIF',
                                    save_all=True,
                                    append_images=combined_frames[1:],
                                    optimize=False,  # Disable optimization to ensure frames remain separate
                                    quality=gif_quality,
                                    duration=300,    # AUMENTADO: 300ms = más lento (mejor para ver detalles)
                                    disposal=2,      # Important: ensures each frame replaces the previous
                                    loop=0           # Loop forever
                                )
                                combined_gif_buffer.seek(0)
                                
                                # Display the combined GIF title
                                st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 30px;'>Preprocessed Image vs Raw Image</div>", unsafe_allow_html=True)
                                
                                # Convert the buffer to base64 to display in HTML
                                import base64
                                combined_gif_buffer.seek(0)
                                gif_base64 = base64.b64encode(combined_gif_buffer.getvalue()).decode()

                                # Display using HTML to force animation
                                html = f"""
                                <div style="display: flex; justify-content: center; width: 100%;">
                                    <img src="data:image/gif;base64,{gif_base64}" style="max-width: 100%;" alt="Steering Angle Animation">
                                </div>
                                """
                                st.markdown(html, unsafe_allow_html=True)

                            else:
                                st.warning("No frames available to create comparison")
                            
                        st.markdown("</div>", unsafe_allow_html=True)

                    else:
                        st.warning("No frames available to create animations")'''

                    st.markdown("</div>", unsafe_allow_html=True)


        else:
            st.session_state.btn = False
    
    with tabs[1]:  # Track Position tab
        st.info("Track Position Analysis - Coming Soon!")
        
    with tabs[2]:  # Driver Behavior tab
        st.info("Driver Behavior Analysis - Coming Soon!")

