import streamlit as st

def show_header():
    st.markdown('''
        <div class="main-header">
            <h1>CSV Q/A ChatBot</h1>
            <p>Upload a CSV file and chat with your data using AI</p>
        </div>
    ''', unsafe_allow_html=True)