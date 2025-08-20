import streamlit as st
import os
import pandas as pd

def sidebar():
    with st.sidebar:
        # Sidebar Header - ensure it's the first element
        st.markdown('<div class="sidebar-header"><h2>⚙️ Functionalities</h2></div>', unsafe_allow_html=True)
        
        # Add some spacing after header
        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 🔑 API Key & LLM Settings")
        st.session_state.llm_type = st.radio(
            "Select LLM Provider",
            ["openai", "gemini", "groq", "huggingface"],
            index=["openai", "gemini", "groq", "huggingface"].index(st.session_state.llm_type) if st.session_state.llm_type in ["openai", "gemini", "groq", "huggingface"] else 0
        )
        llm_type = st.session_state.llm_type
        
        if llm_type == "huggingface":
            st.success("✅ Using HuggingFace (free, open-source) model!")
            st.info("💡 **Note**: Free HuggingFace models may have limitations. For best results, use OpenAI, Gemini, or Groq.")
            st.session_state.hf_model_name = st.text_input(
                "HuggingFace Model Name",
                value=st.session_state.get("hf_model_name", "gpt2"),
                help="Enter the HuggingFace model repo name (e.g., gpt2, distilgpt2, etc.)"
            )
            st.session_state.hf_device = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=["cpu", "cuda"].index(st.session_state.get("hf_device", "cpu")),
                help="Choose 'cpu' for most users, 'cuda' if you have a GPU."
            )
        else:
            if llm_type == "openai":
                api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    st.success(f"✅ {llm_type.capitalize()} API Key set!")
                    st.caption(f"Current Key: {api_key[:4]}...{api_key[-4:]}")
                else:
                    st.warning(f"⚠️ {llm_type.capitalize()} API Key not set")
                st.session_state.openai_api_key = st.text_input(
                    f"{llm_type.capitalize()} API Key",
                    value=api_key or "",
                    type="password",
                    help=f"Enter your {llm_type.capitalize()} API key"
                )
            elif llm_type == "gemini":
                api_key = st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    st.success(f"✅ {llm_type.capitalize()} API Key set!")
                    st.caption(f"Current Key: {api_key[:4]}...{api_key[-4:]}")
                else:
                    st.warning(f"⚠️ {llm_type.capitalize()} API Key not set")
                st.session_state.gemini_api_key = st.text_input(
                    f"{llm_type.capitalize()} API Key",
                    value=api_key or "",
                    type="password",
                    help=f"Enter your {llm_type.capitalize()} API key"
                )
            elif llm_type == "groq":
                api_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
                if api_key:
                    st.success(f"✅ {llm_type.capitalize()} API Key set!")
                    st.caption(f"Current Key: {api_key[:4]}...{api_key[-4:]}")
                else:
                    st.warning(f"⚠️ {llm_type.capitalize()} API Key not set")
                st.session_state.groq_api_key = st.text_input(
                    f"{llm_type.capitalize()} API Key",
                    value=api_key or "",
                    type="password",
                    help=f"Enter your {llm_type.capitalize()} API key"
                )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Chat Settings")
        
        # Strict CSV Mode with visual indicator
        strict_mode = st.checkbox(
            "Strict CSV Mode",
            value=st.session_state.get("strict_csv_mode", False),
            help="When enabled, the AI will only answer questions based on your CSV data"
        )
        st.session_state.strict_csv_mode = strict_mode
        
        # Visual indicator for Strict Mode
        if strict_mode:
            st.markdown('<div class="toggle-indicator toggle-active">🔒 Strict Mode: ACTIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="toggle-indicator toggle-inactive">🔓 Strict Mode: INACTIVE</div>', unsafe_allow_html=True)
        
        # RAG Mode with visual indicator
        rag_mode = st.checkbox(
            "RAG Mode",
            value=st.session_state.get("rag_mode", False),
            help="Enable Retrieval-Augmented Generation for better data-based responses"
        )
        st.session_state.rag_mode = rag_mode
        
        # Visual indicator for RAG Mode
        if rag_mode:
            st.markdown('<div class="toggle-indicator toggle-active">🔍 RAG Mode: ACTIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="toggle-indicator toggle-inactive">🔍 RAG Mode: INACTIVE</div>', unsafe_allow_html=True)
        
        # Advanced RAG Mode (only show if RAG is enabled)
        if rag_mode:
            try:
                from agents_handler.agents import ADVANCED_RAG_AVAILABLE
                if ADVANCED_RAG_AVAILABLE:
                    advanced_rag_mode = st.checkbox(
                        "🚀 Advanced RAG",
                        value=st.session_state.get("advanced_rag_mode", True),
                        help="Enable advanced RAG with semantic search, hybrid retrieval, and enhanced prompting"
                    )
                    st.session_state.advanced_rag_mode = advanced_rag_mode
                    
                    if advanced_rag_mode:
                        st.markdown('<div class="toggle-indicator toggle-active">🚀 Advanced RAG: ACTIVE</div>', unsafe_allow_html=True)
                        st.info("✨ **Advanced Features:**\n• Semantic search with TF-IDF\n• Hybrid retrieval strategies\n• Context-aware prompting\n• Query caching\n• Enhanced metadata analysis")
                    else:
                        st.markdown('<div class="toggle-indicator toggle-inactive">🚀 Advanced RAG: INACTIVE</div>', unsafe_allow_html=True)
                else:
                    st.info("📦 **Advanced RAG:** Install scikit-learn for enhanced features")
            except ImportError:
                st.info("📦 **Advanced RAG:** Install scikit-learn for enhanced features")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Authentication & Chat Management")
        
        # Check if user is logged in
        if 'user' in st.session_state and st.session_state.user:
            user = st.session_state.user
            st.success(f"👤 **Logged in as:** {user['username']}")
            st.caption(f"Provider: {user['auth_provider'].capitalize()}")
            
            # User actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", help="Save current chat session"):
                    from ui_components.auth_system import save_chat_session
                    if 'messages' in st.session_state:
                        save_chat_session(st.session_state.messages)
            
            with col2:
                if st.button("📥 Load Chats", help="Load saved chat sessions"):
                    from ui_components.auth_system import render_chat_management
                    render_chat_management()
            
            if st.button("🚪 Logout", help="Logout from current account"):
                from ui_components.auth_system import logout
                logout()
        else:
            st.info("🔐 **Not signed in**")
            st.caption("Sign in to save and manage your chats")
            
            if st.button("🔐 Sign In", help="Sign in to save chats"):
                from ui_components.auth_system import render_auth_interface
                render_auth_interface()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Help & About")
        st.markdown("""
        **Features:**
        - 📊 CSV Data Analysis
        - 🤖 AI-Powered Q&A
        - 📈 Interactive Visualizations
        - 🔍 RAG (Retrieval-Augmented Generation)
        - 🎯 Multiple LLM Providers
        """)
        
        # Add rate limit info
        if llm_type != "huggingface":
            st.info("💡 **Tip**: If you hit API rate limits, try switching to HuggingFace (free) or another provider in the dropdown above.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Session Info")
        if st.session_state.get("df") is not None:
            df = st.session_state.df
            st.write(f"📁 **File:** {st.session_state.get('filename', 'Unknown')}")
            st.write(f"📊 **Rows:** {len(df)}")
            st.write(f"📋 **Columns:** {len(df.columns)}")
            st.write(f"🤖 **LLM:** {llm_type.capitalize()}")
        else:
            st.write("📁 No file uploaded")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 🧹 Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Data", help="Clear uploaded CSV data"):
                st.session_state.df = None
                st.session_state.messages = []
                st.session_state.filename = None
                st.rerun()
        with col2:
            if st.button("🔄 Reset Session", help="Reset all session data"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📈 KPIs")
        if st.session_state.get("df") is not None:
            df = st.session_state.df
            st.metric("Total Records", len(df))
            st.metric("Total Columns", len(df.columns))
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        else:
            st.write("No data loaded")
        st.markdown('</div>', unsafe_allow_html=True)