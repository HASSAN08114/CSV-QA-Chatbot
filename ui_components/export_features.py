import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import io
import json

def create_download_link(val, filename):
    """Create a download link for files"""
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download {filename}</a>'

def export_dataframe_to_csv(df, filename):
    """Export dataframe to CSV with download link"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" target="_blank">Download CSV</a>'
        return href
    except Exception as e:
        return f"CSV export error: {str(e)}"

def export_dataframe_to_excel(df, filename):
    """Export dataframe to Excel with download link"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" target="_blank">Download Excel</a>'
        return href
    except ImportError:
        return "Excel export not available (openpyxl not installed)"
    except Exception as e:
        return f"Excel export error: {str(e)}"

def export_dataframe_to_json(df, filename):
    """Export dataframe to JSON with download link"""
    try:
        json_data = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_data.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json" target="_blank">Download JSON</a>'
        return href
    except Exception as e:
        return f"JSON export error: {str(e)}"

def export_dataframe_to_pickle(df, filename):
    """Export dataframe to Pickle with download link"""
    try:
        output = io.BytesIO()
        df.to_pickle(output)
        pickle_data = output.getvalue()
        b64 = base64.b64encode(pickle_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pkl" target="_blank">Download Pickle</a>'
        return href
    except Exception as e:
        return f"Pickle export error: {str(e)}"

def generate_analysis_report(df, messages, filename):
    """Generate a comprehensive analysis report"""
    
    # Calculate basic statistics
    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Generate report content
    report_content = f"""
# Data Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** {filename}

## Dataset Overview
- **Total Rows:** {total_rows:,}
- **Total Columns:** {total_cols}
- **Numeric Columns:** {len(numeric_cols)}
- **Categorical Columns:** {len(categorical_cols)}

## Data Quality Assessment
"""
    
    # Add data quality metrics
    missing_data = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    if missing_data > 0:
        report_content += f"- **Missing Values:** {missing_data:,} ({missing_data/(total_rows*total_cols)*100:.1f}%)\n"
    else:
        report_content += "- **Missing Values:** None found\n"
    
    if duplicate_rows > 0:
        report_content += f"- **Duplicate Rows:** {duplicate_rows:,} ({duplicate_rows/total_rows*100:.1f}%)\n"
    else:
        report_content += "- **Duplicate Rows:** None found\n"
    
    # Add chat analysis summary
    if messages:
        report_content += "\n## Chat Analysis Summary\n"
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        report_content += f"- **Total Queries:** {len(user_messages)}\n"
        report_content += f"- **Total Responses:** {len(assistant_messages)}\n"
        
        # Add sample queries
        if user_messages:
            report_content += "\n### Sample Queries:\n"
            for i, msg in enumerate(user_messages[:5], 1):
                report_content += f"{i}. {msg['content'][:100]}...\n"
    
    # Add column statistics
    report_content += "\n## Column Statistics\n"
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        report_content += f"\n### {col}\n"
        report_content += f"- **Type:** {col_type}\n"
        report_content += f"- **Null Values:** {null_count:,}\n"
        report_content += f"- **Unique Values:** {unique_count:,}\n"
        
        if col in numeric_cols:
            try:
                report_content += f"- **Mean:** {df[col].mean():.2f}\n"
                report_content += f"- **Std:** {df[col].std():.2f}\n"
                report_content += f"- **Min:** {df[col].min():.2f}\n"
                report_content += f"- **Max:** {df[col].max():.2f}\n"
            except:
                report_content += "- **Statistics:** Unable to calculate\n"
    
    return report_content

def render_export_features(df, messages, filename):
    """Render export features section"""
    
    st.markdown("### 📤 Export Features")
    
    # Create tabs for different export options
    export_tab1, export_tab2, export_tab3 = st.tabs(["📊 Data Export", "📄 Analysis Report", "🎯 Presentation"])
    
    with export_tab1:
        st.markdown("#### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CSV Export**")
            csv_link = export_dataframe_to_csv(df, filename)
            st.markdown(csv_link, unsafe_allow_html=True)
            
            st.markdown("**Excel Export**")
            excel_link = export_dataframe_to_excel(df, filename)
            st.markdown(excel_link, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**JSON Export**")
            json_link = export_dataframe_to_json(df, filename)
            st.markdown(json_link, unsafe_allow_html=True)
            
            st.markdown("**Pickle Export**")
            pickle_link = export_dataframe_to_pickle(df, filename)
            st.markdown(pickle_link, unsafe_allow_html=True)
    
    with export_tab2:
        st.markdown("#### Generate Analysis Report")
        
        # Generate report
        report_content = generate_analysis_report(df, messages, filename)
        
        # Display report preview
        st.markdown("**Report Preview:**")
        st.text_area("Report Content", report_content, height=400, disabled=True)
        
        # Download report as markdown
        try:
            b64 = base64.b64encode(report_content.encode()).decode()
            report_link = f'<a href="data:text/markdown;base64,{b64}" download="{filename}_report.md" target="_blank">Download Markdown Report</a>'
            st.markdown(report_link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating report download: {str(e)}")
        
        # Download report as text
        try:
            b64 = base64.b64encode(report_content.encode()).decode()
            text_link = f'<a href="data:text/plain;base64,{b64}" download="{filename}_report.txt" target="_blank">Download Text Report</a>'
            st.markdown(text_link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating text download: {str(e)}")
    
    with export_tab3:
        st.markdown("#### Presentation Features")
        
        # Key metrics for presentation
        st.markdown("**Key Metrics for Presentation:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Size", f"{len(df):,} rows")
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            numeric_count = len(df.select_dtypes(include=['number']).columns)
            st.metric("Numeric Columns", numeric_count)
        
        with col4:
            categorical_count = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Columns", categorical_count)
        
        # Generate insights for presentation
        st.markdown("**Presentation Insights:**")
        
        insights = []
        
        # Data size insights
        if len(df) > 10000:
            insights.append("📊 Large dataset with significant sample size")
        elif len(df) > 1000:
            insights.append("📊 Medium-sized dataset suitable for analysis")
        else:
            insights.append("📊 Small dataset - consider collecting more data")
        
        # Data quality insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 5:
            insights.append("✅ High data quality with minimal missing values")
        elif missing_pct < 20:
            insights.append("⚠️ Moderate missing data - consider imputation")
        else:
            insights.append("❌ High missing data - requires data cleaning")
        
        # Column type insights
        if numeric_count > categorical_count:
            insights.append("📈 Numeric-heavy dataset - great for statistical analysis")
        elif categorical_count > numeric_count:
            insights.append("📊 Categorical-heavy dataset - ideal for grouping analysis")
        else:
            insights.append("⚖️ Balanced dataset with mixed data types")
        
        # Display insights
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Export presentation summary
        try:
            presentation_summary = f"""
# {filename} - Presentation Summary

## Key Metrics
- Dataset Size: {len(df):,} rows
- Columns: {len(df.columns)}
- Numeric Columns: {numeric_count}
- Categorical Columns: {categorical_count}

## Insights
{chr(10).join(insights)}

## Analysis Highlights
- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total Queries: {len([msg for msg in messages if msg['role'] == 'user'])}
- Data Quality Score: {calculate_data_quality_score(df)['overall_score']}%
            """
            
            st.markdown("**Download Presentation Summary:**")
            b64 = base64.b64encode(presentation_summary.encode()).decode()
            summary_link = f'<a href="data:text/markdown;base64,{b64}" download="{filename}_presentation.md" target="_blank">Download Presentation Summary</a>'
            st.markdown(summary_link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating presentation summary: {str(e)}")

def calculate_data_quality_score(df):
    """Calculate data quality score (imported from data_preview)"""
    try:
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        uniqueness = 1 - (duplicate_rows / len(df)) if len(df) > 0 else 0
        type_consistency = 1.0
        
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
    except Exception as e:
        return {
            'overall_score': 0.0,
            'completeness': 0.0,
            'uniqueness': 0.0,
            'type_consistency': 0.0,
            'missing_cells': 0,
            'duplicate_rows': 0,
            'total_cells': 0
        }
