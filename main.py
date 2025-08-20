import streamlit as st
import pandas as pd
from io import StringIO
from agents_handler.agents import get_dynamic_agent
import plotly.graph_objects as go
import plotly.express as px
import re
import sys
import io
from custom_css.apply_custom_css import apply_custom_css
import os
import time
from dotenv import load_dotenv
from ui_components.sidebar import sidebar
from ui_components.file_loader import file_uploader_card
from ui_components.extras import show_feature_cards
from ui_components.header import show_header
from ui_components.file_upload import handle_file_upload
from ui_components.data_preview import render_data_preview_tab
from ui_components.chat_analysis import render_chat_analysis_tab
from ui_components.export_features import render_export_features

load_dotenv()

# Apply custom CSS
st.markdown(apply_custom_css(), unsafe_allow_html=True)

# Theme toggle functionality
def add_theme_toggle():
    st.markdown('''
        <button class="theme-toggle" onclick="toggleTheme()" title="Toggle Dark/Light Theme">
            🌙
        </button>
        <script>
            // Initialize theme from localStorage or default to light
            let currentTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', currentTheme);
            updateThemeIcon();
            
            function toggleTheme() {
                currentTheme = currentTheme === 'light' ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', currentTheme);
                localStorage.setItem('theme', currentTheme);
                updateThemeIcon();
            }
            
            function updateThemeIcon() {
                const button = document.querySelector('.theme-toggle');
                if (currentTheme === 'dark') {
                    button.innerHTML = '☀️';
                    button.title = 'Switch to Light Theme';
                } else {
                    button.innerHTML = '🌙';
                    button.title = 'Switch to Dark Theme';
                }
            }
        </script>
    ''', unsafe_allow_html=True)

# Add theme toggle
add_theme_toggle()

# Add sidebar toggle button
st.markdown('''
    <button class="sidebar-toggle" onclick="toggleSidebar()" title="Toggle Sidebar">
        ☰
    </button>
    <script>
        function toggleSidebar() {
            const sidebar = document.querySelector('.css-1d391kg');
            const container = document.querySelector('.block-container');
            
            if (sidebar.classList.contains('sidebar-collapsed')) {
                sidebar.classList.remove('sidebar-collapsed');
                container.classList.remove('content-expanded');
            } else {
                sidebar.classList.add('sidebar-collapsed');
                container.classList.add('content-expanded');
            }
        }
    </script>
''', unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = "gemini"
if 'strict_csv_mode' not in st.session_state:
    st.session_state.strict_csv_mode = False
if 'rag_mode' not in st.session_state:
    st.session_state.rag_mode = False
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'loading' not in st.session_state:
    st.session_state.loading = False

# Show header
show_header()

# Sidebar
sidebar()

# Main content
if st.session_state.df is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "💬 Chat Analysis", "📤 Export", "🔐 Account"])
    
    with tab1:
        render_data_preview_tab(st.session_state.df, st.session_state.filename)
    
    with tab2:
        render_chat_analysis_tab(st.session_state.messages, st.session_state.agent, st.session_state.df)
    
    with tab3:
        render_export_features(st.session_state.df, st.session_state.messages, st.session_state.filename)
    
    with tab4:
        from ui_components.auth_system import render_auth_interface, render_chat_management
        render_auth_interface()
        if 'user' in st.session_state and st.session_state.user:
            st.markdown("---")
            render_chat_management()
else:
    # Create tabs for when no file is uploaded
    tab1, tab2 = st.tabs(["📁 Upload File", "🔐 Account"])
    
    with tab1:
        # File upload section
        st.markdown("### 📁 Upload CSV File")
        handle_file_upload(file_uploader_card)
        
        # Feature highlights with enhanced styling
        st.markdown("### ✨ Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: var(--bg-secondary); border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s ease; box-shadow: var(--shadow-light);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">💬</div>
                <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Natural Language Queries</h4>
                <p style="font-size: 0.875rem; color: var(--text-muted); line-height: 1.5;">Ask questions about your data in plain English with advanced AI understanding</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: var(--bg-secondary); border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s ease; box-shadow: var(--shadow-light);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
                <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">AI-Powered Analysis</h4>
                <p style="font-size: 0.875rem; color: var(--text-muted); line-height: 1.5;">Get intelligent insights with RAG technology and multiple LLM providers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: var(--bg-secondary); border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s ease; box-shadow: var(--shadow-light);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📈</div>
                <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">Interactive Charts</h4>
                <p style="font-size: 0.875rem; color: var(--text-muted); line-height: 1.5;">Generate beautiful visualizations automatically with Plotly integration</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Demo section
        st.markdown("### 🚀 Quick Demo")
        demo_col1, demo_col2 = st.columns([2, 1])
        
        with demo_col1:
            st.markdown("""
            <div style="background: var(--bg-secondary); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color);">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">Try These Sample Queries:</h4>
                <ul style="color: var(--text-secondary); line-height: 1.8;">
                    <li>"Show me a bar chart of sales by region"</li>
                    <li>"What are the top 5 products by revenue?"</li>
                    <li>"Calculate the average age of customers"</li>
                    <li>"Find correlations between price and sales volume"</li>
                    <li>"Create a scatter plot of price vs rating"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with demo_col2:
            st.markdown("""
            <div style="background: var(--info-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--info-border);">
                <h4 style="color: var(--info-text); margin-bottom: 1rem;">💡 Pro Tips</h4>
                <ul style="color: var(--info-text); line-height: 1.6; font-size: 0.875rem;">
                    <li>Enable RAG mode for better accuracy</li>
                    <li>Use strict CSV mode to prevent hallucinations</li>
                    <li>Try different LLM providers for varied responses</li>
                    <li>Save your chat sessions for later reference</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        from ui_components.auth_system import render_auth_interface, render_chat_management
        render_auth_interface()
        if 'user' in st.session_state and st.session_state.user:
            st.markdown("---")
            render_chat_management()