import streamlit as st
import pandas as pd
from typing import Tuple
import plotly.graph_objects as go


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
        file_name="Steering_Angle_Results.csv",
        mime="text/csv"
    )
    st.markdown("")

def create_line_chart(df: pd.DataFrame):
    """Create a line chart with the given DataFrame"""
    fig = go.Figure()

    # Add the main steering angle line
    fig.add_trace(go.Scatter(
        x=df['time'], 
        y=df['steering_angle'],
        mode='lines',
        name='Steering Angle',
        line=dict(color='white', width=1),
        hovertemplate='<b>Time:</b> %{x}<br><b>Angle:</b> %{y:.2f}°<extra></extra>'
    ))

    # Add reference lines for straight, full right, and full left
    fig.add_shape(type="line",
        x0=df['time'].min(), y0=0, x1=df['time'].max(), y1=0,
        line=dict(color="red", width=2, dash="solid"),
        name="Straight (0°)"
    )

    fig.add_shape(type="line",
        x0=df['time'].min(), y0=90, x1=df['time'].max(), y1=90,
        line=dict(color="red", width=2, dash="dash"),
        name="Full Right (90°)"
    )

    fig.add_shape(type="line",
        x0=df['time'].min(), y0=-90, x1=df['time'].max(), y1=-90,
        line=dict(color="red", width=2, dash="dash"),
        name="Full Left (-90°)"
    )

    # Añadir etiquetas a las líneas de referencia
    fig.add_annotation(x=df['time'].min(), y=0,
        text="Straight (0°)",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=-20
    )

    fig.add_annotation(x=df['time'].min(), y=90,
        text="Full Right (90°)",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=-20
    )

    fig.add_annotation(x=df['time'].min(), y=-90,
        text="Full Left (-90°)",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=20
    )

    # Configure layout
    fig.update_layout(
        title="Steering Angle Over Time",
        xaxis_title="Time (seconds)",
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
        x0=df['time'].min(), y0=-10,
        x1=df['time'].max(), y1=10,
        fillcolor="lightgray",
        opacity=0.2,
        layer="below",
        line_width=0,
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)