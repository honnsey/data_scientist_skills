import streamlit as st
import numpy as np
import pandas as pd

def app():
    """
    Page shows all jobs for browsing through
    """

    st.write('''We can provide you with a number of jobs that are best suited to your skills.
             Please tell us a bit about yourself and your experience!''')
    st.write('''Please tick the boxes below to describe your skills.''')

    # List of skills
    skill_list = ['Excellent Communication Skills', 'create', 'Data Analysis', 'develop',
                'development', 'Data Engineering', 'information', 'Project Management',
                'ML Model Build', 'Python Programming', 'science', 'software', 'SQL', 'systems']


    # User input and data collection
    input_collector = {}
    columns = st.columns(2)

    for skill in skill_list[:7]:
        if columns[0].checkbox(skill):
            input_collector[skill] = 1
        else:
            input_collector[skill] = 0

    for skill in skill_list[7:]:
        if columns[1].checkbox(skill):
            input_collector[skill] = 1
        else:
            input_collector[skill] = 0

    st.slider('Years of Experience in Data Science',1, 20, 5)

    if st.checkbox("Happy with your descriptions above?"):
        if sum(input_collector.values()) > 0:
            st.success('Below are the top 5 jobs that we recommend you to apply!')
                ## display dataframe of recommended jobs below

        else:
            st.info('Would you be interested in some data science courses to improve your skill set?')
