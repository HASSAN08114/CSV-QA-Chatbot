import json
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

class TrainingDataCollector:
    """
    Collects training data from user interactions in the CSV Q/A ChatBot
    """
    
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.session_file = os.path.join(data_dir, "session_data.jsonl")
        self.training_file = os.path.join(data_dir, "training_data.jsonl")
        
    def collect_from_session(self, messages: List[Dict], df: pd.DataFrame = None):
        """
        Collect training data from current session messages
        """
        if not messages or len(messages) < 2:
            return
        
        # Get CSV context
        csv_context = self._create_csv_context(df) if df is not None else ""
        
        # Extract Q&A pairs from messages
        qa_pairs = self._extract_qa_pairs(messages, csv_context)
        
        # Save to session file
        self._save_session_data(qa_pairs)
        
        return qa_pairs
    
    def _create_csv_context(self, df: pd.DataFrame) -> str:
        """Create context string from DataFrame"""
        if df is None or df.empty:
            return ""
        
        context = f"""
CSV Dataset Information:
- Shape: {df.shape}
- Columns: {', '.join(df.columns.tolist())}
- Data Types: {dict(df.dtypes)}
- Missing Values: {df.isnull().sum().to_dict()}

Sample Data:
{df.head().to_string()}
"""
        return context
    
    def _extract_qa_pairs(self, messages: List[Dict], csv_context: str) -> List[Dict]:
        """Extract Q&A pairs from chat messages"""
        qa_pairs = []
        
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                    # Create training example
                    training_example = {
                        "timestamp": datetime.now().isoformat(),
                        "input_text": f"{csv_context}\n\nQuestion: {user_msg['content']}",
                        "target_text": assistant_msg['content'],
                        "metadata": {
                            "question_type": self._classify_question(user_msg['content']),
                            "response_type": self._classify_response(assistant_msg['content']),
                            "csv_shape": self._get_csv_shape(csv_context)
                        }
                    }
                    qa_pairs.append(training_example)
        
        return qa_pairs
    
    def _classify_question(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['plot', 'chart', 'graph', 'visualize', 'show']):
            return "visualization"
        elif any(word in question_lower for word in ['count', 'how many', 'number', 'total']):
            return "counting"
        elif any(word in question_lower for word in ['average', 'mean', 'median', 'statistics', 'correlation']):
            return "statistics"
        elif any(word in question_lower for word in ['find', 'search', 'where', 'which']):
            return "search"
        else:
            return "general"
    
    def _classify_response(self, response: str) -> str:
        """Classify response type"""
        response_lower = response.lower()
        
        if "```python" in response or "plotly" in response_lower:
            return "code_generation"
        elif any(word in response_lower for word in ['chart', 'graph', 'plot']):
            return "visualization"
        else:
            return "text_response"
    
    def _get_csv_shape(self, csv_context: str) -> str:
        """Extract CSV shape from context"""
        if "Shape:" in csv_context:
            shape_line = [line for line in csv_context.split('\n') if 'Shape:' in line][0]
            return shape_line.split('Shape:')[1].strip()
        return "unknown"
    
    def _save_session_data(self, qa_pairs: List[Dict]):
        """Save session data to file"""
        with open(self.session_file, 'a', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + '\n')
    
    def prepare_training_data(self, min_samples: int = 50) -> List[Dict]:
        """Prepare training data from collected sessions"""
        training_data = []
        
        # Read all session data
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        training_data.append(json.loads(line))
        
        # Filter and clean data
        cleaned_data = self._clean_training_data(training_data)
        
        # Save to training file
        with open(self.training_file, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item) + '\n')
        
        return cleaned_data
    
    def _clean_training_data(self, data: List[Dict]) -> List[Dict]:
        """Clean and filter training data"""
        cleaned = []
        
        for item in data:
            # Filter out very short responses
            if len(item['target_text']) < 10:
                continue
            
            # Filter out very long inputs (likely too complex)
            if len(item['input_text']) > 2000:
                continue
            
            # Remove any responses with errors
            if 'error' in item['target_text'].lower():
                continue
            
            cleaned.append(item)
        
        return cleaned
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        if not os.path.exists(self.training_file):
            return {"total_samples": 0, "question_types": {}, "response_types": {}}
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        question_types = {}
        response_types = {}
        
        for item in data:
            q_type = item['metadata']['question_type']
            r_type = item['metadata']['response_type']
            
            question_types[q_type] = question_types.get(q_type, 0) + 1
            response_types[r_type] = response_types.get(r_type, 0) + 1
        
        return {
            "total_samples": len(data),
            "question_types": question_types,
            "response_types": response_types
        }

# Streamlit integration
def add_data_collection_to_chat():
    """Add data collection to the existing chat interface"""
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = TrainingDataCollector()
    
    # Collect data after each chat interaction
    if 'messages' in st.session_state and 'df' in st.session_state:
        if len(st.session_state.messages) > 0:
            qa_pairs = st.session_state.data_collector.collect_from_session(
                st.session_state.messages, 
                st.session_state.df
            )
            
            # Show collection status in sidebar
            if qa_pairs:
                st.sidebar.success(f"📊 Collected {len(qa_pairs)} new training examples")
