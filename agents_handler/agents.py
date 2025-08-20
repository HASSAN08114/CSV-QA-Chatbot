import os
from langchain_community.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

import pandas as pd
from typing import Optional, Any, List
import re
from langchain.llms.base import LLM
from pydantic import Field

# Add HuggingFace support
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

# Add advanced RAG support
try:
    from .advanced_rag import create_advanced_rag_agent, AdvancedRAGSystem
    from .prompt_engineer import create_enhanced_prompt_engineer, AdvancedPromptEngineer
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False
    print("Warning: Advanced RAG features not available. Install scikit-learn for enhanced features.")

class HuggingFaceLLM(LLM):
    model_name: str = Field(default="gpt2")
    device: Optional[str] = Field(default=None)
    generator: Any = Field(default=None)
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None, **kwargs):
        if pipeline is None:
            raise ImportError("transformers is not installed. Please install with 'pip install transformers'.")
        
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.generator = pipeline("text-generation", model=model_name, tokenizer=model_name, device=device, max_new_tokens=100)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self.generator(prompt, return_full_text=False, max_length=len(prompt.split()) + 50)
        return result[0]["generated_text"] if result and "generated_text" in result[0] else str(result)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def invoke(self, prompt, **kwargs):
        # Handle both string prompts and message objects
        if hasattr(prompt, 'content'):
            prompt = prompt.content
        elif isinstance(prompt, list) and len(prompt) > 0:
            # Handle list of messages
            if hasattr(prompt[0], 'content'):
                prompt = prompt[0].content
            else:
                prompt = str(prompt[0])
        return self._call(prompt)

def get_huggingface_llm(model_name: str = "gpt2", device: Optional[str] = None):
    return HuggingFaceLLM(model_name=model_name, device=device)

def get_dynamic_agent(df, llm_type="openai", temperature=0, api_key=None, hf_model_name=None, hf_device=None):
    """
    Returns a LangChain agent for a given DataFrame using the selected LLM.
    llm_type: "openai", "gemini", "groq", or "huggingface"
    api_key: Optional. If not provided, will check Streamlit session_state, then environment variable.
    hf_model_name: Optional. HuggingFace model name if using huggingface.
    hf_device: Optional. Device for HuggingFace model (e.g., "cpu" or "cuda").
    """
    if llm_type == "openai":
        key = api_key or st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key not set. Please enter your key in the sidebar.")
        llm = OpenAI(model="gpt-4o-mini", temperature=temperature, api_key=key)
    elif llm_type == "gemini":
        key = api_key or st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google Gemini API key not set. Please enter your key in the sidebar.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, api_key=key)
    elif llm_type == "groq":
        key = api_key or st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API key not set. Please enter your key in the sidebar.")
        llm = ChatGroq(model="llama3-70b-8192", temperature=temperature, groq_api_key=key)
    elif llm_type == "huggingface":
        model_name = hf_model_name or st.session_state.get("hf_model_name") or "gpt2"
        device = hf_device or st.session_state.get("hf_device")
        try:
            llm = get_huggingface_llm(model_name, device)
        except Exception as e:
            # Fallback to a simpler model if the requested one fails
            print(f"Warning: Failed to load {model_name}, falling back to gpt2: {e}")
            llm = get_huggingface_llm("gpt2", device)
    else:
        raise ValueError("Unsupported LLM type. Choose 'openai', 'gemini', 'groq', or 'huggingface'.")
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
    )
    return agent

def simple_text_search(df, query, top_k=5):
    """
    Improved text-based search to find relevant rows from DataFrame.
    This implements RAG concepts with better search capabilities.
    """
    query_lower = query.lower()
    relevant_rows = []
    scores = []
    
    # Extract key terms from query
    query_terms = set(query_lower.split())
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
    query_terms = query_terms - stop_words
    
    # Search through each row
    for idx, row in df.iterrows():
        score = 0
        row_text = ""
        
        # Combine all column values into searchable text
        for col in df.columns:
            value = str(row[col]).lower()
            row_text += value + " "
        
        # Calculate relevance score
        for term in query_terms:
            if term in row_text:
                score += 1
                # Bonus for exact matches
                if term in row_text.split():
                    score += 0.5
                # Bonus for column name matches
                for col in df.columns:
                    if term in col.lower():
                        score += 0.3
        
        # Check for partial matches and synonyms
        if 'nust' in query_lower and 'nust' in row_text:
            score += 2
        if 'entry' in query_lower and 'entry' in row_text:
            score += 1
        if 'test' in query_lower and 'test' in row_text:
            score += 1
        if 'attempt' in query_lower and ('attempt' in row_text or 'time' in row_text or 'number' in row_text):
            score += 1.5
        
        if score > 0:
            relevant_rows.append(row)
            scores.append(score)
    
    # Sort by relevance score
    if relevant_rows:
        sorted_pairs = sorted(zip(relevant_rows, scores), key=lambda x: x[1], reverse=True)
        relevant_rows = [row for row, score in sorted_pairs[:top_k]]
    
    # If no exact matches, try broader search
    if not relevant_rows:
        for idx, row in df.iterrows():
            row_text = str(row.to_dict()).lower()
            # Check if any word from query appears in the row
            if any(word in row_text for word in query_terms):
                relevant_rows.append(row)
                if len(relevant_rows) >= top_k:
                    break
    
    # If still no matches, return first few rows
    if not relevant_rows:
        relevant_rows = df.head(top_k).to_dict('records')
    
    return relevant_rows

def get_rag_agent(df, llm, top_k=5):
    """
    Returns a RAG agent for a given DataFrame using improved text search.
    This demonstrates RAG concepts with better retrieval and prompting.
    """
    def rag_query(user_query):
        # Retrieve relevant rows using improved text search
        relevant_rows = simple_text_search(df, user_query, top_k)
        
        # Convert to context string with better formatting
        context_parts = []
        for i, row in enumerate(relevant_rows, 1):
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            context_parts.append(f"Row {i}: {row_text}")
        
        context = "\n".join(context_parts)
        
        # Limit context size
        max_context_chars = 1500
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n... [context truncated] ..."
        
        # Create improved RAG prompt
        prompt = f"""
You are a helpful data assistant using Retrieval-Augmented Generation (RAG). 
The user has uploaded a CSV file, and I've found the following most relevant rows for their question:

{context}

**Instructions:**
1. Answer the user's question using ONLY the information in these rows
2. If the answer is clearly present in the data, provide it directly
3. If the answer is partially present, provide what you can find
4. If the answer cannot be found in these rows, say "The provided data does not contain information about [specific aspect of the question]"
5. Be specific and accurate - don't make assumptions

**User Question:** {user_query}

**Available Columns:** {', '.join([str(c) for c in df.columns])}

Please provide a clear, direct answer based on the retrieved data.
"""
        response = llm.invoke(prompt)
        
        # Import the comprehensive cleaning function
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ui_components'))
        try:
            from response_cleaner import clean_llm_response
            # Use the comprehensive cleaning function
            return clean_llm_response(response)
        except ImportError:
            # Fallback cleaning if response_cleaner is not available
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
    
    return rag_query

def get_advanced_rag_agent(df, llm, top_k=5):
    """
    Returns an advanced RAG agent with multiple retrieval strategies and enhanced prompting.
    This provides superior performance compared to the basic RAG system.
    """
    if not ADVANCED_RAG_AVAILABLE:
        print("Advanced RAG not available, falling back to basic RAG")
        return get_rag_agent(df, llm, top_k)
    
    try:
        return create_advanced_rag_agent(df, llm, top_k)
    except Exception as e:
        print(f"Error creating advanced RAG agent: {e}")
        print("Falling back to basic RAG agent")
        return get_rag_agent(df, llm, top_k)

def get_enhanced_dynamic_agent(df, llm_type="openai", temperature=0, api_key=None, 
                              hf_model_name=None, hf_device=None, use_advanced_rag=True):
    """
    Enhanced dynamic agent that can use advanced RAG when available.
    """
    # Get the LLM
    if llm_type == "openai":
        key = api_key or st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key not set. Please enter your key in the sidebar.")
        llm = OpenAI(model="gpt-4o-mini", temperature=temperature, api_key=key)
    elif llm_type == "gemini":
        key = api_key or st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google Gemini API key not set. Please enter your key in the sidebar.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, api_key=key)
    elif llm_type == "groq":
        key = api_key or st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API key not set. Please enter your key in the sidebar.")
        llm = ChatGroq(model="llama3-70b-8192", temperature=temperature, groq_api_key=key)
    elif llm_type == "huggingface":
        model_name = hf_model_name or st.session_state.get("hf_model_name") or "gpt2"
        device = hf_device or st.session_state.get("hf_device")
        try:
            llm = get_huggingface_llm(model_name, device)
        except Exception as e:
            # Fallback to a simpler model if the requested one fails
            print(f"Warning: Failed to load {model_name}, falling back to gpt2: {e}")
            llm = get_huggingface_llm("gpt2", device)
    else:
        raise ValueError("Unsupported LLM type. Choose 'openai', 'gemini', 'groq', or 'huggingface'.")
    
    # Create agent based on preference
    if use_advanced_rag and ADVANCED_RAG_AVAILABLE:
        return get_advanced_rag_agent(df, llm)
    else:
        return get_rag_agent(df, llm)