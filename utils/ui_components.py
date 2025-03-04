import streamlit as st
import pandas as pd
from typing import Tuple

def create_header():
    """Create the application header"""
    st.markdown("""
        <div class='custom-header'>
            F1 Video Analysis Platform
            <div style='font-size: 0.5em; font-weight: 400; margin-top: 10px;'>
                Precision Telemetry & Analysis
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_upload_section():
    """Create the video upload section"""
    st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload onboard camera footage for analysis"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file

def create_frame_selector(total_frames: int) -> Tuple[int, int]:
    """Create frame selection controls with slider and +/- buttons"""
    st.markdown("<div class='glassmorphic-container'>", unsafe_allow_html=True)


    # Create a slider for frame range selection
    start_frame, end_frame = st.select_slider(
        "Select Frame Range",
        options=range(0, total_frames),
        value=(0, total_frames-1),
        format_func=lambda x: f"Frame {x}"
    )
    
    


    st.markdown("</div>", unsafe_allow_html=True)
    return start_frame, end_frame

def display_results(df: pd.DataFrame):


    csv = df.to_csv(index=False)
    st.markdown("")


    st.markdown("#### Download Results 📥")

    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="f1_analysis_results.csv",
        mime="text/csv"
    )
    st.markdown("")
