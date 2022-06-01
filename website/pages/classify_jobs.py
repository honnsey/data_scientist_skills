import streamlit as st
import numpy as np
import pandas as pd

def app():
    """
    Page shows all jobs for browsing through
    """
    st.markdown('''Are you looking for a new employee in the data science field?
                There are many different roles in the field of data science, and we
                can help you categorise what you're looking for!
             ''')
    st.write("You can choose one of the two options below")

    columns = st.columns(2)
    if columns[0].button('Enter a Job Description'):
        title = st.text_input('Job Description','Enter Job Description Here')
    # title will then be the input for model prediction

    if columns[1].button('Upload a Job Description'):
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write(data)
