#streamlit run your_script.py
import streamlit as st

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

#st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)




st.markdown("<br>",unsafe_allow_html=True)

pages = st.navigation({ 
    "Models": [
        st.Page("pages/steering-angle.py", title="Steering Angle Prediction"),
        st.Page("pages/soon.py", title="Soon...")],})

pages.run()