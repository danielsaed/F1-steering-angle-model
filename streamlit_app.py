#streamlit run your_script.py
import streamlit as st
import os
import sys
from utils.helper import BASE_DIR,metrics_page
from pathlib import Path

st.set_page_config(
        page_title="F1 Video Analysis Platform",
        page_icon="🏎️",
        initial_sidebar_state="expanded",
        layout="wide"
    )

if "visited" not in st.session_state:
    st.session_state["visited"] = True
    try:
        metrics_page.update_one({"page": "inicio"}, {"$inc": {"visits": 1}})
    except:
        st.warning("MongoDB client not connected.")


hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
logo_style = '''
    <style>
        /* Estilo para iconos de contacto - versión compacta */
        .contact-icons {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .contact-icon {
            display: flex;
            align-items: center;
            padding: 6px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            text-decoration: none;
            color: #ffffff;
        }
        
        .contact-icon:hover {
            background-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .contact-icon img {
            width: 16px;
            height: 16px;
        }
        
        /* Email button style */
        .email-button {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px 15px;
            border-radius: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            text-decoration: none;
            color: #ffffff;
            font-size: 13px;
            margin-top: 12px;
            width: 100%;
        }
        
        .email-button:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        
        .email-button img {
            width: 16px;
            height: 16px;
            margin-right: 8px;
        }
        
        /* Estilo para el separador */
        .sidebar-separator {
            margin: 20px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
'''
#sst.markdown(hide_decoration_bar_style, unsafe_allow_html=True)logo_style
st.markdown(logo_style, unsafe_allow_html=True)

#st.markdown("<br>",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: #fff;'>Considerations</h3>", unsafe_allow_html=True)
    st.caption("""**Ouput Data**:""")
    st.markdown("<p style='text-align: left; color: gray; font-size: 12px;'>The model is trained on over 22,000 images from -180° to 180° with a potential 2.5° error, for the moment may not accurately predict angles beyond 180°. Poor or high-intensity lighting may affect data accuracy.</p>", unsafe_allow_html=True)
    st.caption("""**Usage**:""") 
    st.markdown("<p style='text-align: left; color: gray; font-size: 12px;'>Free-tier server resources are limited, so the page may be slow or crash with large files. To run it locally, feel free to fork/clone the project or download the desktop app.</p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: left; color: gray; font-size: 12px;'>Any feedback is welcome.</p>", unsafe_allow_html=True)
    st.markdown("", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #fff;'>Contact</h3>", unsafe_allow_html=True)
    # Nueva versión más compacta de los iconos
    contact_html = """
    <div class="contact-icons">
        <a href='https://x.com/aaa' target="_blank" class="contact-icon" title="X">
            <img src="https://static.vecteezy.com/system/resources/previews/053/986/348/non_2x/x-twitter-icon-logo-symbol-free-png.png" alt="X">
        </a>
        <a href="https://github.com/danielsaed/F1-steering-angle-model" target="_blank" class="contact-icon" title="GitHub">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
        </a>
    </div>
    
    """
    
    st.markdown(contact_html, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.markdown("<p style='text-align: center; color: gray; font-size: 10px;'>For research/educational purposes only</p>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("")

    st.markdown("<h3 style='text-align: center; color: #fff;'>Get Desktop App</h3>", unsafe_allow_html=True)
    col1,col2, col3 = st.columns([1,6,1])
    with col2:

        st.markdown("<p style='text-align: center; color: gray; font-size: 10px;'>Click Assets then download .exe</p>", unsafe_allow_html=True)
        st.link_button("Download", "https://github.com/danielsaed/F1-steering-angle-model/releases",type="secondary",use_container_width=True)


pages = st.navigation({ 
    "Steering Angle Model": [
        st.Page(Path(BASE_DIR) / "navigation" / "steering-angle.py", title="Use Model"),
        st.Page(Path(BASE_DIR) / "navigation" / "soon.py", title="Historical Steering Data Base"),
        ],})

pages.run()

