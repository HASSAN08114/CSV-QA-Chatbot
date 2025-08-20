import streamlit as st

def file_uploader_card():
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your data file in CSV format"
    )
    return uploaded_file