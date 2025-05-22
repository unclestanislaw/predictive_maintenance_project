import streamlit as st
from analysis_and_model import analysis_and_model_page 
from presentation import presentation_page 

pages = {
    "": [
        st.Page("analysis_and_model.py", title="Анализ и модель"),
        st.Page("presentation.py", title="Презентация"),
    ]
}


pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
