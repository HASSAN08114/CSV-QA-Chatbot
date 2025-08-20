import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def calculate_data_quality_score(df):
    """Calculate a comprehensive data quality score"""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Calculate various quality metrics
    completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
    uniqueness = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 0
    
    # Check for data type consistency
    type_consistency = 1.0
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric columns are stored as objects
            try:
                pd.to_numeric(df[col], errors='raise')
                type_consistency -= 0.1
            except:
                pass
    
    # Overall quality score
    quality_score = (completeness * 0.4 + uniqueness * 0.3 + type_consistency * 0.3) * 100
    
    return {
        'overall_score': round(quality_score, 1),
        'completeness': round(completeness * 100, 1),
        'uniqueness': round(uniqueness * 100, 1),
        'type_consistency': round(type_consistency * 100, 1),
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
        'total_cells': total_cells
    }

def create_data_overview_chart(df):
    """Create a comprehensive data overview chart"""
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    
    # Missing values per column
    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Types Distribution', 'Missing Values by Column', 
                       'Dataset Shape', 'Data Quality Metrics'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Data types pie chart
    fig.add_trace(
        go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, 
               name="Data Types", hole=0.4),
        row=1, col=1
    )
    
    # Missing values bar chart
    if len(missing_per_col) > 0:
        fig.add_trace(
            go.Bar(x=missing_per_col.index, y=missing_per_col.values, 
                   name="Missing Values", marker_color='#ef4444'),
            row=1, col=2
        )
    
    # Dataset shape indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=len(df),
            title={"text": "Total Rows"},
            delta={"reference": 0},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=1
    )
    
    # Data quality score
    quality_metrics = calculate_data_quality_score(df)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=quality_metrics['overall_score'],
            title={'text': "Data Quality Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}]},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, 
                     title_text="Dataset Overview", title_x=0.5)
    return fig

def render_data_preview_tab(df, filename):
    """Enhanced data preview with analytics and visualizations"""
    
    # Header with file info
    st.markdown(f"### 📊 Data Preview: {filename}")
    
    # Data quality assessment
    quality_metrics = calculate_data_quality_score(df)
    
    # Quality score display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Quality Score", f"{quality_metrics['overall_score']}%")
    
    with col2:
        st.metric("Completeness", f"{quality_metrics['completeness']}%")
    
    with col3:
        st.metric("Uniqueness", f"{quality_metrics['uniqueness']}%")
    
    with col4:
        st.metric("Type Consistency", f"{quality_metrics['type_consistency']}%")
    
    # Quality alerts
    if quality_metrics['overall_score'] < 80:
        st.warning(f"⚠️ Data quality score is {quality_metrics['overall_score']}%. Consider data cleaning for better analysis.")
    
    if quality_metrics['missing_cells'] > 0:
        st.info(f"ℹ️ Found {quality_metrics['missing_cells']} missing values. Use 'Fill missing values' or 'Remove rows with missing data' in chat.")
    
    if quality_metrics['duplicate_rows'] > 0:
        st.info(f"ℹ️ Found {quality_metrics['duplicate_rows']} duplicate rows. Consider removing duplicates for cleaner analysis.")
    
    # Data overview chart
    st.markdown("### 📈 Dataset Overview")
    overview_chart = create_data_overview_chart(df)
    st.plotly_chart(overview_chart, use_container_width=True)
    
    # Basic statistics
    st.markdown("### 📋 Basic Statistics")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.markdown("#### Numeric Columns")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            st.markdown("#### Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Columns")
        
        # Show unique values and their counts for each categorical column
        for col in categorical_cols[:5]:  # Limit to first 5 columns to avoid clutter
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Only show if reasonable number of unique values
                st.markdown(f"**{col}** ({unique_count} unique values)")
                
                # Create bar chart for top values
                value_counts = df[col].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values,
                    title=f"Top 10 values in {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("### 🔍 Data Preview")
    
    # Show first few rows
    st.markdown("#### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Show last few rows
    st.markdown("#### Last 10 Rows")
    st.dataframe(df.tail(10), use_container_width=True)
    
    # Column information
    st.markdown("### 📝 Column Information")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Unique Values': df[col].nunique(),
            'Memory Usage': df[col].memory_usage(deep=True)
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)
    
    # Data insights
    st.markdown("### 💡 Quick Insights")
    
    insights = []
    
    # Dataset size insights
    insights.append(f"📊 Dataset contains {len(df):,} rows and {len(df.columns)} columns")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    insights.append(f"💾 Memory usage: {memory_usage / 1024:.1f} KB")
    
    # Missing data insights
    if quality_metrics['missing_cells'] > 0:
        missing_percentage = (quality_metrics['missing_cells'] / quality_metrics['total_cells']) * 100
        insights.append(f"⚠️ {missing_percentage:.1f}% of data is missing")
    
    # Duplicate insights
    if quality_metrics['duplicate_rows'] > 0:
        duplicate_percentage = (quality_metrics['duplicate_rows'] / len(df)) * 100
        insights.append(f"🔄 {duplicate_percentage:.1f}% of rows are duplicates")
    
    # Data type insights
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = len(df.select_dtypes(include=['object']).columns)
    insights.append(f"🔢 {numeric_count} numeric columns, {categorical_count} categorical columns")
    
    # Display insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    # Recommendations
    st.markdown("### 🎯 Recommendations")
    
    recommendations = []
    
    if quality_metrics['overall_score'] < 80:
        recommendations.append("🔧 Consider data cleaning to improve quality score")
    
    if quality_metrics['missing_cells'] > 0:
        recommendations.append("🧹 Handle missing values using imputation or removal")
    
    if quality_metrics['duplicate_rows'] > 0:
        recommendations.append("🗑️ Remove duplicate rows for cleaner analysis")
    
    if len(numeric_cols) > 0:
        recommendations.append("📈 Use numeric columns for statistical analysis and visualizations")
    
    if len(categorical_cols) > 0:
        recommendations.append("📊 Use categorical columns for grouping and segmentation analysis")
    
    if not recommendations:
        recommendations.append("✅ Your data looks good! Ready for analysis.")
    
    for rec in recommendations:
        st.markdown(f"- {rec}") 