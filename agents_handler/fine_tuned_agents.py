import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from typing import Optional, Dict, Any
import streamlit as st

class FineTunedCSVAgent:
    """
    Fine-tuned agent for CSV Q&A using QLoRA
    """
    
    def __init__(self, base_model_path: str, adapter_path: str, device: str = "auto"):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the fine-tuned model"""
        if self.is_loaded:
            return
        
        try:
            print("Loading fine-tuned model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()
            
            self.is_loaded = True
            print("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            self.is_loaded = False
    
    def generate_response(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """Generate response using fine-tuned model"""
        if not self.is_loaded:
            self.load_model()
        
        if not self.is_loaded:
            return "Error: Fine-tuned model not loaded"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def batch_generate(self, prompts: list, batch_size: int = 4) -> list:
        """Generate responses for multiple prompts"""
        if not self.is_loaded:
            self.load_model()
        
        if not self.is_loaded:
            return ["Error: Fine-tuned model not loaded"] * len(prompts)
        
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Move to device
                if self.device != "auto":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate responses
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode responses
                batch_responses = []
                for j, output in enumerate(outputs):
                    response = self.tokenizer.decode(output, skip_special_tokens=True)
                    response = response.replace(batch_prompts[j], "").strip()
                    batch_responses.append(response)
                
                responses.extend(batch_responses)
                
            except Exception as e:
                print(f"Error in batch generation: {e}")
                responses.extend([f"Error: {str(e)}"] * len(batch_prompts))
        
        return responses

def get_fine_tuned_agent(model_path: str = "./fine_tuned_model") -> Optional[FineTunedCSVAgent]:
    """
    Get fine-tuned agent if available
    """
    if not os.path.exists(model_path):
        return None
    
    # Check if model files exist
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    if not all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
        return None
    
    try:
        # Try to create fine-tuned agent
        agent = FineTunedCSVAgent(
            base_model_path="microsoft/DialoGPT-medium",  # Base model
            adapter_path=model_path,  # LoRA adapter
            device="auto"
        )
        
        # Test loading
        agent.load_model()
        
        if agent.is_loaded:
            return agent
        else:
            return None
            
    except Exception as e:
        print(f"Error creating fine-tuned agent: {e}")
        return None

# Integration with existing agents system
def get_enhanced_rag_agent(df, llm, fine_tuned_agent=None, top_k=5):
    """
    Enhanced RAG agent that can use fine-tuned model if available
    """
    def enhanced_rag_query(user_query):
        # Retrieve relevant rows using improved text search
        from agents_handler.agents import simple_text_search
        relevant_rows = simple_text_search(df, user_query, top_k)
        
        # Convert to context string
        context_parts = []
        for i, row in enumerate(relevant_rows, 1):
            row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            context_parts.append(f"Row {i}: {row_text}")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""
Context: You are analyzing a CSV dataset with the following relevant rows:

{context}

Available Columns: {', '.join([str(c) for c in df.columns])}

Question: {user_query}

Please provide a clear, accurate answer based on the data above.
"""
        
        # Use fine-tuned model if available, otherwise use base model
        if fine_tuned_agent and fine_tuned_agent.is_loaded:
            response = fine_tuned_agent.generate_response(prompt)
        else:
            response = llm.invoke(prompt)
            
            # Clean response if it's from base model
            from ui_components.response_cleaner import clean_llm_response
            response = clean_llm_response(response)
        
        return response
    
    return enhanced_rag_query

# Streamlit integration
def add_fine_tuned_model_to_sidebar():
    """Add fine-tuned model controls to sidebar"""
    st.sidebar.subheader("🤖 Fine-Tuned Model")
    
    # Check if fine-tuned model exists
    model_path = "./fine_tuned_model"
    model_exists = os.path.exists(model_path) and any(
        os.path.exists(os.path.join(model_path, f)) 
        for f in ["config.json", "pytorch_model.bin"]
    )
    
    if model_exists:
        st.sidebar.success("✅ Fine-tuned model available")
        
        # Toggle for using fine-tuned model
        use_fine_tuned = st.sidebar.checkbox(
            "Use Fine-Tuned Model", 
            value=True,
            help="Use the custom fine-tuned model for enhanced performance"
        )
        
        if use_fine_tuned:
            st.sidebar.info("🎯 Using fine-tuned model for better CSV understanding")
            
            # Model info
            with st.sidebar.expander("Model Info"):
                st.write("**Model Type:** QLoRA Fine-tuned")
                st.write("**Base Model:** DialoGPT-medium")
                st.write("**Training:** CSV Q&A specific")
                st.write("**Features:** Enhanced data understanding")
        
        return use_fine_tuned
    else:
        st.sidebar.warning("⚠️ No fine-tuned model found")
        st.sidebar.info("💡 Train a model using the training interface")
        return False
