import streamlit as st
import numpy as np
import pandas as pd
from data_scientist_skills.data import get_data

def app():
    """
    Page shows all jobs for browsing through
    """
    @st.cache
    def get_cached_data():
        return get_data()

    st.markdown('''
            Welcome to Data2$$$ Job Search - a job search engine curated for the Data Science Field.

             We have thousands of job listings available, so have a look!
             ''')
    st.write(get_cached_data().iloc[:,:3])
