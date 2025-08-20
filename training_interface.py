#!/usr/bin/env python3
"""
Quick Training Interface for CSV Q&A Fine-tuning
Run this script to train a fine-tuned model on collected data
"""

import streamlit as st
import os
import json
import pandas as pd
from training.data_collector import TrainingDataCollector
from training.qlora_trainer import quick_train_from_collected_data
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="CSV Q&A Model Training",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 CSV Q&A Model Training Interface")
    st.markdown("Train a fine-tuned model on your collected data")
    
    # Initialize data collector
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = TrainingDataCollector()
    
    # Sidebar
    st.sidebar.header("Training Controls")
    
    # Data collection status
    st.sidebar.subheader("📊 Data Collection")
    stats = st.session_state.data_collector.get_data_stats()
    
    st.sidebar.metric("Total Samples", stats["total_samples"])
    
    if stats["total_samples"] > 0:
        st.sidebar.success("✅ Data collected successfully!")
        
        # Show question type distribution
        if stats["question_types"]:
            st.sidebar.subheader("Question Types")
            for q_type, count in stats["question_types"].items():
                st.sidebar.metric(q_type.title(), count)
    else:
        st.sidebar.warning("⚠️ No training data collected yet")
        st.sidebar.info("💡 Use the main app to collect data first")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Training Data Analysis")
        
        if stats["total_samples"] > 0:
            # Load training data for analysis
            training_file = "training_data/training_data.jsonl"
            if os.path.exists(training_file):
                with open(training_file, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                
                # Create analysis
                df_analysis = pd.DataFrame([
                    {
                        'question_type': item['metadata']['question_type'],
                        'response_type': item['metadata']['response_type'],
                        'input_length': len(item['input_text']),
                        'output_length': len(item['target_text'])
                    }
                    for item in data
                ])
                
                # Question type distribution
                fig1 = px.pie(
                    df_analysis, 
                    names='question_type', 
                    title="Question Type Distribution"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Response type distribution
                fig2 = px.bar(
                    df_analysis['response_type'].value_counts().reset_index(),
                    x='index',
                    y='response_type',
                    title="Response Type Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Length analysis
                fig3 = px.scatter(
                    df_analysis,
                    x='input_length',
                    y='output_length',
                    color='question_type',
                    title="Input vs Output Length"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
            else:
                st.warning("Training data file not found")
        else:
            st.info("No training data available. Use the main app to collect data first.")
    
    with col2:
        st.subheader("🚀 Model Training")
        
        if stats["total_samples"] >= 20:
            st.success(f"✅ Ready to train! ({stats['total_samples']} samples)")
            
            # Training parameters
            st.subheader("Training Parameters")
            
            model_name = st.selectbox(
                "Base Model",
                ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-small", "microsoft/DialoGPT-large"],
                help="Choose the base model for fine-tuning"
            )
            
            num_epochs = st.slider("Training Epochs", 1, 10, 3)
            learning_rate = st.selectbox(
                "Learning Rate",
                ["1e-4", "2e-4", "5e-4", "1e-3"],
                index=1
            )
            
            # Training button
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    try:
                        # Update training parameters
                        os.environ["LEARNING_RATE"] = learning_rate
                        
                        # Start training
                        result = quick_train_from_collected_data(
                            model_name=model_name,
                            min_samples=20
                        )
                        
                        if result:
                            st.success("🎉 Training completed successfully!")
                            st.balloons()
                            
                            # Show model info
                            st.subheader("📋 Model Information")
                            st.write(f"**Base Model:** {model_name}")
                            st.write(f"**Training Epochs:** {num_epochs}")
                            st.write(f"**Learning Rate:** {learning_rate}")
                            st.write(f"**Training Samples:** {stats['total_samples']}")
                            
                            st.info("💡 The fine-tuned model is now available in the main app!")
                        else:
                            st.error("❌ Training failed. Check the console for details.")
                            
                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
        
        elif stats["total_samples"] > 0:
            st.warning(f"⚠️ Need at least 20 samples (current: {stats['total_samples']})")
            st.info("💡 Continue using the main app to collect more data")
        else:
            st.warning("⚠️ No training data available")
            st.info("💡 Use the main app to collect data first")
    
    # Data management
    st.subheader("🗂️ Data Management")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("📊 View Sample Data"):
            if stats["total_samples"] > 0:
                training_file = "training_data/training_data.jsonl"
                if os.path.exists(training_file):
                    with open(training_file, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f if line.strip()]
                    
                    # Show sample
                    sample = data[0]
                    st.json(sample)
    
    with col4:
        if st.button("🧹 Clean Data"):
            if stats["total_samples"] > 0:
                cleaned_data = st.session_state.data_collector.prepare_training_data()
                st.success(f"✅ Data cleaned! {len(cleaned_data)} samples ready for training")
    
    with col5:
        if st.button("🗑️ Clear All Data"):
            if st.checkbox("I understand this will delete all collected data"):
                try:
                    os.remove("training_data/session_data.jsonl")
                    os.remove("training_data/training_data.jsonl")
                    st.success("✅ All data cleared!")
                    st.rerun()
                except:
                    st.error("❌ Error clearing data")

if __name__ == "__main__":
    main()
