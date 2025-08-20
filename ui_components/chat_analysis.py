# If you see import errors, make sure to install these packages:
# pip install streamlit plotly langchain-core
import streamlit as st
import re
import time
import sys
import io
import plotly.graph_objects as go
import plotly.express as px
import datetime
try:
    from langchain_core.messages import HumanMessage
except ImportError:
    HumanMessage = None  # Add fallback or error message if needed
from agents_handler.agents import OpenAI, ChatGoogleGenerativeAI, ChatGroq, get_rag_agent
import os

def clean_llm_response(response):
    """
    Comprehensive function to clean LLM responses and remove all metadata.
    Handles different response formats from various LLM providers.
    """
    if response is None:
        return ""
    
    # Extract content from different response object types
    if hasattr(response, 'content'):
        clean_response = response.content
    elif hasattr(response, 'text'):
        clean_response = response.text
    elif hasattr(response, 'response'):
        clean_response = response.response
    elif hasattr(response, 'message') and hasattr(response.message, 'content'):
        clean_response = response.message.content
    elif isinstance(response, str):
        clean_response = response
    else:
        clean_response = str(response)
    
    # If it's a string with metadata, try to extract just the content
    if isinstance(clean_response, str):
        # Try to extract content from various patterns
        content_patterns = [
            r"content='([^']*)'",
            r'content="([^"]*)"',
            r"content=([^,\s}]*)",
            r"text='([^']*)'",
            r'text="([^"]*)"',
            r"text=([^,\s}]*)"
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, clean_response)
            if match:
                clean_response = match.group(1)
                break
    
    # First, try to extract the actual answer from dictionary-like responses
    if isinstance(clean_response, str):
        # Look for the specific pattern: {'input': '...', 'output': '...'}
        dict_pattern = r"\{[^}]*'input':[^}]*'output':\s*'([^']*)'[^}]*\}"
        dict_match = re.search(dict_pattern, clean_response)
        if dict_match:
            clean_response = dict_match.group(1)
            return clean_response.strip()
        
        # Also try with double quotes
        dict_pattern2 = r"\{[^}]*\"input\":[^}]*\"output\":\s*\"([^\"]*)\"[^}]*\}"
        dict_match2 = re.search(dict_pattern2, clean_response)
        if dict_match2:
            clean_response = dict_match2.group(1)
            return clean_response.strip()
    
    # Comprehensive metadata removal
    metadata_patterns = [
        r'additional_kwargs=\{.*?\}',
        r'response_metadata=\{.*?\}',
        r'usage_metadata=\{.*?\}',
        r'id=\'[^\']*\'',
        r'finish_reason=\'[^\']*\'',
        r'model_name=\'[^\']*\'',
        r'system_fingerprint=\'[^\']*\'',
        r'service_tier=\'[^\']*\'',
        r'logprobs=\'[^\']*\'',
        r'Available columns:.*',
        r'content=\'',
        r'content="',
        r'text=\'',
        r'text="',
        r'generation_info=\{.*?\}',
        r'prompt_feedback=\{.*?\}',
        r'candidates=\[.*?\]',
        r'usage_metadata=\{.*?\}',
        r'token_usage=\{.*?\}',
        r'completion_tokens=\d+',
        r'prompt_tokens=\d+',
        r'total_tokens=\d+',
        r'completion_time=[\d.]+',
        r'prompt_time=[\d.]+',
        r'queue_time=[\d.]+',
        r'total_time=[\d.]+',
        r'run_id=\'[^\']*\'',
        r'run_type=\'[^\']*\'',
        r'start_time=[\d.]+',
        r'end_time=[\d.]+',
        r'extra=\{[^}]*\}',
        r'name=\'[^\']*\'',
        r'type=\'[^\']*\'',
        r'data=\{[^}]*\}',
        r'is_chunk=\w+',
        r'generation=\{[^}]*\}',
        r'usage=\{[^}]*\}',
        r'prompt_tokens=\d+',
        r'completion_tokens=\d+',
        r'total_tokens=\d+',
        # Remove specific backend patterns
        r'General analysis: \{.*?\}',
        r'RAG: \{.*?\}',
        r'Strict CSV: \{.*?\}'
    ]
    
    for pattern in metadata_patterns:
        clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining curly braces and their content
    clean_response = re.sub(r'\{[^}]*\}', '', clean_response)
    
    # Remove any remaining square brackets and their content
    clean_response = re.sub(r'\[[^\]]*\]', '', clean_response)
    
    # Remove any remaining parentheses with metadata-like content
    clean_response = re.sub(r'\([^)]*metadata[^)]*\)', '', clean_response, flags=re.IGNORECASE)
    
    # Clean up extra whitespace, newlines, and punctuation
    clean_response = re.sub(r'\s+', ' ', clean_response)
    clean_response = re.sub(r'^\s*[\'"]\s*', '', clean_response)  # Remove leading quotes
    clean_response = re.sub(r'\s*[\'"]\s*$', '', clean_response)  # Remove trailing quotes
    clean_response = re.sub(r'^\s*,\s*', '', clean_response)  # Remove leading commas
    clean_response = re.sub(r'\s*,\s*$', '', clean_response)  # Remove trailing commas
    
    return clean_response.strip()

def is_data_question(prompt, df):
    """
    Check if a question is specifically about the uploaded CSV data.
    Only return True if the question clearly references column names or data-specific terms.
    """
    # Only check for column names, not generic data terms
    columns = [str(col).lower() for col in df.columns]
    prompt_lower = prompt.lower()
    
    # Check if any column name is mentioned in the question
    if any(col in prompt_lower for col in columns):
        return True
    
    # Check for specific data analysis terms that clearly relate to the CSV
    data_specific_keywords = ['plot', 'chart', 'visualization', 'graph', 'show me', 'display']
    if any(keyword in prompt_lower for keyword in data_specific_keywords):
        return True
    
    return False

def get_llm(llm_type, api_key=None):
    if llm_type == "openai":
        key = api_key or st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        return OpenAI(model="gpt-4o-mini", temperature=0, api_key=key)
    elif llm_type == "gemini":
        key = api_key or st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=key)
    elif llm_type == "groq":
        key = api_key or st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        return ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=key)
    elif llm_type == "huggingface":
        from agents_handler.agents import get_huggingface_llm
        model_name = st.session_state.get("hf_model_name", "gpt2")
        device = st.session_state.get("hf_device", "cpu")
        return get_huggingface_llm(model_name, device)
    else:
        raise ValueError("Unsupported LLM type.")

def render_chat_analysis_tab(messages, agent, df):
    # Use columns to create a more compact layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Removed empty wrapper that created extra spacing in the DOM
        
        # Show welcome message if no messages - make it more compact
        if not messages:
            st.markdown('''
                <div class="welcome-message" style="margin-bottom: 20px;">
                    <h3>🤖 Welcome to CSV Q/A ChatBot</h3>
                    <p>Upload a CSV file and start chatting with your data. Ask questions, request visualizations, or get insights!</p>
                </div>
            ''', unsafe_allow_html=True)
        
        # Display messages in a more compact format
        if messages:
            # Create a container for messages with limited height
            message_container = st.container()
            with message_container:
                for idx, message in enumerate(messages):
                    timestamp = message.get("timestamp")
                    if not timestamp:
                        timestamp = datetime.datetime.now().strftime("%H:%M")
                    
                    # Determine avatar and role based on message type
                    if message["role"] == "user":
                        avatar = "👤"
                        role = "You"
                        message_class = "user-message"
                    elif message["role"] == "assistant":
                        avatar = "🤖"
                        role = "AI Assistant"
                        message_class = "bot-message"
                    elif message["role"] == "plot":
                        avatar = "📊"
                        role = "Chart"
                        message_class = "bot-message"
                    else:
                        avatar = "ℹ️"
                        role = "Info"
                        message_class = "bot-message"
                    
                    st.markdown(f'''
                        <div class="chat-message {message_class}" style="margin-bottom: 10px;">
                            <div class="message-header">
                                <div class="message-avatar">{avatar}</div>
                                <div class="message-role">{role}</div>
                            </div>
                            <div class="message-content">{message["content"]}</div>
                            <div class="message-timestamp">{timestamp}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    if message["role"] == "plot":
                        st.plotly_chart(message["content"], key=f"plotly_{idx}", use_container_width=True)
        
        # Chat input at the bottom - properly positioned
        st.markdown('<div style="margin-top: 20px; position: sticky; bottom: 0; background: white; padding: 10px; border-top: 1px solid #e0e0e0; z-index: 1000;">', unsafe_allow_html=True)
        if prompt := st.chat_input("Ask a question about your data or anything...", key="chat_input"):
            now = datetime.datetime.now().strftime("%H:%M")
            messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": now
            })
            st.markdown(f'''
                <div class="chat-message user-message" style="margin-bottom: 10px;">
                    <div class="message-header">
                        <div class="message-avatar">👤</div>
                        <div class="message-role">You</div>
                    </div>
                    <div class="message-content">{prompt}</div>
                    <div class="message-timestamp">{now}</div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Process the response
            with st.spinner("🤖 Processing your question..."):
                llm_type = st.session_state.llm_type
                api_key = None
                if llm_type == "openai":
                    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                elif llm_type == "gemini":
                    api_key = st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
                elif llm_type == "groq":
                    api_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
                
                # Check for advanced RAG
                rag_mode = st.session_state.get('rag_mode', False)
                advanced_rag_mode = st.session_state.get('advanced_rag_mode', False)
                
                try:
                    strict_mode = st.session_state.get("strict_csv_mode", False)
                    
                    # Handle Strict CSV Mode first
                    if strict_mode and df is not None and not df.empty:
                        try:
                            from agents_handler.agents import get_dynamic_agent
                            agent = get_dynamic_agent(df, llm_type, temperature=0, api_key=api_key)
                            response = agent.invoke(prompt)
                            label = "Strict CSV: "
                        except Exception as strict_error:
                            response = f"Error in strict mode: {str(strict_error)}"
                            label = "Strict CSV Error: "
                    
                    # Handle RAG Mode
                    elif rag_mode and df is not None and not df.empty:
                        try:
                            if advanced_rag_mode:
                                from agents_handler.agents import get_advanced_rag_agent
                                llm = get_llm(llm_type, api_key)
                                rag_agent = get_advanced_rag_agent(df, llm)
                                response = rag_agent(prompt)
                                label = "Advanced RAG: "
                            else:
                                llm = get_llm(llm_type, api_key)
                                rag_agent = get_rag_agent(df, llm)
                                response = rag_agent(prompt)
                                label = "RAG: "
                        except Exception as rag_error:
                            response = f"Error in RAG mode: {str(rag_error)}"
                            label = "RAG Error: "
                    
                    # Handle General Mode
                    else:
                        llm = get_llm(llm_type, api_key)
                        
                        # Check if the question is about dataset statistics or data analysis
                        data_keywords = [
                            'rows', 'columns', 'shape', 'size', 'dimensions', 'dataset', 'data',
                            'how many', 'count', 'length', 'number of', 'total', 'records',
                            'missing', 'null', 'empty', 'duplicate', 'unique', 'data types',
                            'top', 'best', 'highest', 'lowest', 'average', 'mean', 'sum',
                            'revenue', 'sales', 'products', 'category', 'region', 'amount',
                            'units', 'price', 'rating', 'satisfaction', 'marketing', 'profit'
                        ]
                        
                        is_data_related = any(keyword in prompt.lower() for keyword in data_keywords)
                        
                        if is_data_related and df is not None:
                            # For data questions, use the pandas DataFrame agent for better results
                            try:
                                from agents_handler.agents import get_dynamic_agent
                                agent = get_dynamic_agent(df, llm_type, temperature=0.1, api_key=api_key)
                                response = agent.invoke(prompt)
                                label = "General analysis: "
                            except Exception as agent_error:
                                # Fallback to enhanced prompt if agent fails
                                dataset_info = f"""
You are analyzing a CSV dataset with the following information:

Dataset Overview:
- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
- Total cells: {df.shape[0] * df.shape[1]:,}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
- Missing values: {df.isnull().sum().sum():,}
- Duplicate rows: {df.duplicated().sum():,}

Column Information:
- Column names: {', '.join(df.columns.tolist())}
- Data types: {dict(df.dtypes.value_counts())}
- Numeric columns: {len(df.select_dtypes(include=['number']).columns)}
- Categorical columns: {len(df.select_dtypes(include=['object']).columns)}
- Datetime columns: {len(df.select_dtypes(include=['datetime']).columns)}

Sample Data (first 5 rows):
{df.head().to_string()}

User Question: {prompt}

Please provide a detailed answer based on the dataset information above. If the question requires calculations or analysis, provide the specific values and insights from the data.
"""
                                enhanced_prompt = dataset_info
                                try:
                                    if llm_type == "huggingface":
                                        response = llm.invoke(enhanced_prompt)
                                    elif HumanMessage is not None:
                                        response = llm.invoke([HumanMessage(content=enhanced_prompt)])
                                    else:
                                        response = llm.invoke(enhanced_prompt)
                                except Exception as e:
                                    raise e
                        else:
                            # For non-data questions, use direct LLM call
                            try:
                                if llm_type == "huggingface":
                                    response = llm.invoke(enhanced_prompt)
                                elif HumanMessage is not None:
                                    response = llm.invoke([HumanMessage(content=enhanced_prompt)])
                                else:
                                    response = llm.invoke(enhanced_prompt)
                            except Exception as e:
                                error_msg = str(e)
                                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "429" in error_msg:
                                    messages.append({
                                        "role": "assistant",
                                        "content": f"⚠️ **API Rate Limit Reached**: The {llm_type.capitalize()} API has reached its quota limit. Please try:\n\n1. **Switch to a different LLM** (OpenAI, Groq, or HuggingFace)\n2. **Wait a few minutes** and try again\n3. **Check your API plan** and billing details\n\nYou can change the LLM provider in the sidebar.",
                                        "timestamp": now
                                    })
                                else:
                                    messages.append({
                                        "role": "assistant",
                                        "content": f"❌ **Error**: {error_msg}",
                                        "timestamp": now
                                    })
                                return
                        
                        # Use the existing comprehensive response cleaning function
                        clean_response = clean_llm_response(response)
                        
                        # Ensure label is always defined
                        if 'label' not in locals():
                            label = "General knowledge: "
                    
                    time.sleep(0.5)
                    
                    # Clean the response
                    clean_response = clean_llm_response(response)
                    response_str = clean_response if clean_response else str(response)
                    
                    # Store the response for immediate display
                    final_response = clean_response if clean_response else str(response)
                    
                    # Add the response to messages and display it immediately
                    assistant_message = {"role": "assistant", "content": f"{label}{final_response}", "timestamp": now}
                    messages.append(assistant_message)
                    
                    # Display the response immediately
                    st.markdown(f'''
                        <div class="chat-message bot-message" style="margin-bottom: 10px;">
                            <div class="message-header">
                                <div class="message-avatar">🤖</div>
                                <div class="message-role">AI Assistant</div>
                            </div>
                            <div class="message-content">{assistant_message["content"]}</div>
                            <div class="message-timestamp">{assistant_message["timestamp"]}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                except Exception as e:
                    messages.append({
                        "role": "assistant",
                        "content": f"❌ **Error**: {str(e)}",
                        "timestamp": now
                    })
        # Close sticky input wrapper only (avoid extra blank container)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right sidebar for quick actions
    with col2:
        st.markdown("### Quick Actions")
        if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 Save Chat", help="Save current chat session"):
            save_chat_session(messages)
        
        if st.button("📥 Load Chat", help="Load a saved chat session"):
            load_chat_session()
        
        st.markdown("---")
        st.markdown("### Chat Stats")
        if messages:
            st.write(f"📝 **Messages**: {len(messages)}")
            user_messages = len([m for m in messages if m['role'] == 'user'])
            st.write(f"👤 **Your Questions**: {user_messages}")
            assistant_messages = len([m for m in messages if m['role'] == 'assistant'])
            st.write(f"🤖 **AI Responses**: {assistant_messages}")
        else:
            st.write("📝 **Messages**: 0")
            st.write("👤 **Your Questions**: 0")
            st.write("🤖 **AI Responses**: 0") 
