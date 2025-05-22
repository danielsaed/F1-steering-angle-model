#streamlit run your_script.py
import streamlit as st
import os
import sys
from utils.helper import BASE_DIR
from pathlib import Path

st.set_page_config(
        page_title="F1 Video Analysis Platform",
        page_icon="🏎️",
        initial_sidebar_state="expanded",
        layout="wide"
    )


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
    st.caption("""**Usage**:""") 
    st.markdown("<p style='text-align: left; color: gray; font-size: 12px;'>The app will likely crash or will be too slow if there is high traffic; If you use the app frequently I recomend you to fork or clone it and run it locally, it will take 3-5 min.</p>", unsafe_allow_html=True)
    # 
    st.caption(str(Path(BASE_DIR) / "navigation" / "soon.py"))
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
    st.markdown("<p style='text-align: center; color: gray; font-size: 10px;'>For research/educational purposes only, its not related to F1 or any organization.</p>", unsafe_allow_html=True)


pages = st.navigation({ 
    "Steering Angle Model": [
        st.Page(Path(BASE_DIR) / "navigation" / "steering-angle.py", title="Use Model"),
        st.Page(Path(BASE_DIR) / "navigation" / "soon.py", title="Help To Improve Model"),
        ],})

pages.run()

