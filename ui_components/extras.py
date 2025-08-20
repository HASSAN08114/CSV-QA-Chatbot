import streamlit as st

def show_feature_cards():
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💬 Natural Language Queries")
        st.markdown("Ask questions about your data in plain English")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI-Powered Analysis")
        st.markdown("Leverage OpenAI, Google Gemini, or Groq for insights")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Visualization")
        st.markdown("Generate charts and graphs from your data")
        st.markdown('</div>', unsafe_allow_html=True)