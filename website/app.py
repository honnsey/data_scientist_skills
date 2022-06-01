import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import browse_jobs, recommend_jobs, classify_jobs

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("From Data to $$$")

# Add all your applications (pages) here
app.add_page("Job Search", browse_jobs.app)
app.add_page("Job Recommendation", recommend_jobs.app)
app.add_page("Classify Job", classify_jobs.app)


# The main app
app.run()
