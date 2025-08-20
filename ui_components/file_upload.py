import streamlit as st
import pandas as pd
from agents_handler.agents import get_dynamic_agent
import os

def create_dataset_summary(df):
    """
    Creates a comprehensive dataset summary with statistics
    """
    summary = {
        "shape": df.shape,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.value_counts().to_dict(),
        "column_names": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns),
        "datetime_columns": list(df.select_dtypes(include=['datetime']).columns)
    }
    return summary

def display_dataset_info(df, filename):
    """
    Displays comprehensive dataset information in a beautiful format
    """
    summary = create_dataset_summary(df)
    
    # Create a beautiful dataset info display
    st.markdown("### 📊 Dataset Information")
    
    # Main statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Rows", 
            value=f"{summary['rows']:,}",
            help="Total number of data rows"
        )
    
    with col2:
        st.metric(
            label="📋 Columns", 
            value=f"{summary['columns']}",
            help="Total number of columns"
        )
    
    with col3:
        st.metric(
            label="💾 Memory", 
            value=f"{summary['memory_usage'] / 1024:.1f} KB",
            help="Memory usage of the dataset"
        )
    
    with col4:
        st.metric(
            label="❓ Missing", 
            value=f"{summary['missing_values']:,}",
            help="Total missing values"
        )
    
    # Additional statistics
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric(
            label="🔄 Duplicates", 
            value=f"{summary['duplicate_rows']:,}",
            help="Number of duplicate rows"
        )
    
    with col6:
        missing_percentage = (summary['missing_values'] / (summary['rows'] * summary['columns'])) * 100
        st.metric(
            label="📊 Missing %", 
            value=f"{missing_percentage:.1f}%",
            help="Percentage of missing values"
        )
    
    # Data types breakdown
    st.markdown("#### 📝 Data Types")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.markdown(f"**Numeric Columns:** {len(summary['numeric_columns'])}")
        if summary['numeric_columns']:
            st.write(", ".join(summary['numeric_columns'][:3]))
            if len(summary['numeric_columns']) > 3:
                st.write(f"... and {len(summary['numeric_columns']) - 3} more")
    
    with col8:
        st.markdown(f"**Categorical Columns:** {len(summary['categorical_columns'])}")
        if summary['categorical_columns']:
            st.write(", ".join(summary['categorical_columns'][:3]))
            if len(summary['categorical_columns']) > 3:
                st.write(f"... and {len(summary['categorical_columns']) - 3} more")
    
    with col9:
        st.markdown(f"**Datetime Columns:** {len(summary['datetime_columns'])}")
        if summary['datetime_columns']:
            st.write(", ".join(summary['datetime_columns']))
    
    # Column names
    st.markdown("#### 🏷️ Column Names")
    st.write(", ".join(summary['column_names']))
    
    # Success message with enhanced information
    st.success(f"""
    ✅ **Successfully loaded:** {filename}
    
    📊 **Dataset Shape:** {summary['shape'][0]:,} rows × {summary['shape'][1]} columns
    
    💡 **Ready for analysis!** You can now ask questions about your data, request visualizations, or perform data analysis.
    """)

def handle_file_upload(file_uploader_card):
    """
    Handles file upload, DataFrame creation, and session state updates.
    Returns True if a file was successfully uploaded and processed, else False.
    """
    llm_type = st.session_state.llm_type
    api_key = None
    
    if llm_type == "openai":
        api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    elif llm_type == "gemini":
        api_key = st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
    elif llm_type == "groq":
        api_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    elif llm_type == "huggingface":
        # HuggingFace doesn't need an API key
        api_key = "no_key_needed"

    # Only check for API key if it's not HuggingFace
    if llm_type != "huggingface" and not api_key:
        st.warning(f"Please enter your {llm_type.capitalize()} API key in the sidebar before uploading a CSV file.")
        st.file_uploader(
            "Choose a CSV file (API key required)",
            type="csv",
            help="Enter your API key in the sidebar first.",
            disabled=True
        )
        return False

    uploaded_file = file_uploader_card()
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.filename = uploaded_file.name
            st.session_state.agent = get_dynamic_agent(df, llm_type=llm_type, api_key=api_key)
            
            # Display comprehensive dataset information
            display_dataset_info(df, uploaded_file.name)
            
            # Enhanced welcome message with dataset statistics
            summary = create_dataset_summary(df)
            welcome_message = f"""
            Hello! I've successfully loaded your CSV file **'{uploaded_file.name}'** 📊
            
            **Dataset Overview:**
            • **Shape:** {summary['shape'][0]:,} rows × {summary['shape'][1]} columns
            • **Memory Usage:** {summary['memory_usage'] / 1024:.1f} KB
            • **Missing Values:** {summary['missing_values']:,} ({summary['missing_values'] / (summary['shape'][0] * summary['shape'][1]) * 100:.1f}%)
            • **Data Types:** {len(summary['numeric_columns'])} numeric, {len(summary['categorical_columns'])} categorical
            
            **What you can do:**
            • Ask questions about your data (e.g., "How many rows are there?")
            • Request visualizations (e.g., "Show me a bar chart of column X")
            • Perform analysis (e.g., "What's the correlation between A and B?")
            • Filter data (e.g., "Show rows where column X > 100")
            
            What would you like to explore? 🚀
            """
            
            st.session_state.messages = [{
                "role": "assistant",
                "content": welcome_message
            }]
            
            st.rerun()
            return True
        except ImportError as e:
            st.error(f"Import error: {str(e)}")
            st.session_state.df = pd.DataFrame()
            st.session_state.filename = None
            st.session_state.agent = None
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.session_state.df = pd.DataFrame()
            st.session_state.filename = None
            st.session_state.agent = None
            st.session_state.messages = []
    return False 