import pandas as pd
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

class AdvancedPromptEngineer:
    """
    Advanced prompt engineering system for RAG applications
    """
    
    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()
        self.few_shot_examples = self._load_few_shot_examples()
        self.context_optimizers = self._load_context_optimizers()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load different prompt templates for various query types"""
        return {
            'statistical': """
You are an expert data analyst specializing in statistical analysis. You have access to a dataset with the following characteristics:

{dataset_info}

**Context from RAG System:**
{context}

**User Question:** {query}

**Instructions for Statistical Analysis:**
1. Identify the relevant columns for the statistical question
2. Perform the appropriate statistical calculation
3. Provide interpretation of the results
4. Include confidence intervals or significance levels when applicable
5. Suggest additional analyses if relevant

**Available Columns:** {columns}

Please provide a comprehensive statistical analysis with clear explanations.
""",
            
            'filtering': """
You are a data filtering expert. You need to help the user find specific data based on their criteria.

**Dataset Information:**
{dataset_info}

**Context from RAG System:**
{context}

**User Question:** {query}

**Instructions for Data Filtering:**
1. Identify the filtering criteria from the user's question
2. Apply appropriate filters to the dataset
3. Present the filtered results clearly
4. Provide summary statistics of the filtered data
5. Suggest additional filters if relevant

**Available Columns:** {columns}

Please provide the filtered data and relevant insights.
""",
            
            'aggregation': """
You are a data aggregation specialist. You need to help the user group and summarize data.

**Dataset Information:**
{dataset_info}

**Context from RAG System:**
{context}

**User Question:** {query}

**Instructions for Data Aggregation:**
1. Identify the grouping columns and aggregation functions
2. Perform the aggregation operations
3. Present results in a clear, organized manner
4. Provide insights about the aggregated data
5. Suggest visualizations if appropriate

**Available Columns:** {columns}

Please provide comprehensive aggregation results with insights.
""",
            
            'visualization': """
You are a data visualization expert. You need to create meaningful charts and graphs.

**Dataset Information:**
{dataset_info}

**Context from RAG System:**
{context}

**User Question:** {query}

**Instructions for Visualization:**
1. Identify the best chart type for the data and question
2. Select appropriate columns for x-axis, y-axis, and grouping
3. Create clear, informative visualizations
4. Provide interpretation of the visualizations
5. Suggest additional visualizations if helpful

**Available Columns:** {columns}

Please create the requested visualization with clear explanations.
""",
            
            'general': """
You are an intelligent data assistant with expertise in data analysis and interpretation.

**Dataset Information:**
{dataset_info}

**Context from RAG System:**
{context}

**User Question:** {query}

**Instructions:**
1. Answer the question using the provided context
2. Provide clear, accurate information
3. Include relevant insights and observations
4. Suggest follow-up questions if appropriate
5. Use the dataset metadata to provide context

**Available Columns:** {columns}

Please provide a comprehensive answer based on the available data.
"""
        }
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot learning examples for different query types"""
        return {
            'statistical': [
                {
                    'question': 'What is the average salary in the dataset?',
                    'context': 'Dataset has salary column with numeric values',
                    'answer': 'The average salary in the dataset is $65,420. This represents the mean of all salary values across the entire dataset.'
                },
                {
                    'question': 'How many missing values are there in each column?',
                    'context': 'Dataset has missing values in various columns',
                    'answer': 'Missing values by column:\n- Name: 0 missing values\n- Age: 12 missing values (2.4%)\n- Salary: 8 missing values (1.6%)\n- Department: 3 missing values (0.6%)'
                }
            ],
            'filtering': [
                {
                    'question': 'Show me all employees in the Engineering department',
                    'context': 'Dataset has department column with Engineering values',
                    'answer': 'Found 45 employees in the Engineering department. Here are the key statistics:\n- Average salary: $85,420\n- Age range: 24-58 years\n- Most common role: Software Engineer'
                }
            ]
        }
    
    def _load_context_optimizers(self) -> Dict[str, callable]:
        """Load context optimization functions"""
        return {
            'truncate': self._truncate_context,
            'summarize': self._summarize_context,
            'prioritize': self._prioritize_context,
            'structure': self._structure_context
        }
    
    def _truncate_context(self, context: str, max_length: int = 2000) -> str:
        """Truncate context to fit within token limits"""
        if len(context) <= max_length:
            return context
        
        # Try to keep the most important parts
        lines = context.split('\n')
        important_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > max_length:
                break
            important_lines.append(line)
            current_length += len(line) + 1
        
        return '\n'.join(important_lines) + '\n... [context truncated] ...'
    
    def _summarize_context(self, context: str) -> str:
        """Create a summary of the context"""
        lines = context.split('\n')
        summary_lines = []
        
        # Extract key information
        for line in lines:
            if any(keyword in line.lower() for keyword in ['shape:', 'columns:', 'rows:', 'missing:', 'data types:']):
                summary_lines.append(line)
        
        return '\n'.join(summary_lines)
    
    def _prioritize_context(self, context: str, query: str) -> str:
        """Prioritize context based on query relevance"""
        query_terms = set(query.lower().split())
        lines = context.split('\n')
        prioritized_lines = []
        
        for line in lines:
            relevance_score = sum(1 for term in query_terms if term in line.lower())
            if relevance_score > 0:
                prioritized_lines.insert(0, line)  # Put relevant lines first
            else:
                prioritized_lines.append(line)
        
        return '\n'.join(prioritized_lines)
    
    def _structure_context(self, context: str) -> str:
        """Structure context for better readability"""
        sections = {
            'Dataset Overview': [],
            'Relevant Data': [],
            'Metadata': []
        }
        
        lines = context.split('\n')
        current_section = 'Dataset Overview'
        
        for line in lines:
            if 'Dataset Overview:' in line:
                current_section = 'Dataset Overview'
            elif 'Relevant Data Rows:' in line:
                current_section = 'Relevant Data'
            elif any(keyword in line for keyword in ['Columns:', 'Data Types:', 'Missing Values:']):
                current_section = 'Metadata'
            
            sections[current_section].append(line)
        
        # Reconstruct with better structure
        structured_context = []
        for section_name, section_lines in sections.items():
            if section_lines:
                structured_context.append(f"\n**{section_name}**")
                structured_context.extend(section_lines)
        
        return '\n'.join(structured_context)
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and type of the query"""
        query_lower = query.lower()
        
        # Intent classification
        intents = {
            'statistical': ['average', 'mean', 'median', 'sum', 'count', 'total', 'percentage', 'correlation', 'standard deviation', 'variance'],
            'filtering': ['where', 'filter', 'show', 'find', 'select', 'rows where', 'data where', 'only', 'just'],
            'aggregation': ['group by', 'grouped', 'by category', 'by department', 'by type', 'summarize', 'summary'],
            'visualization': ['plot', 'chart', 'graph', 'visualize', 'show me a', 'create a', 'bar chart', 'scatter plot', 'histogram'],
            'exploration': ['what is', 'tell me about', 'describe', 'explain', 'how many', 'what are'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'between', 'higher', 'lower', 'better', 'worse']
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Query complexity
        complexity = 'simple'
        if len(query.split()) > 10 or any(intent in detected_intents for intent in ['statistical', 'aggregation']):
            complexity = 'complex'
        
        # Data requirements
        data_requirements = []
        if any(word in query_lower for word in ['numeric', 'number', 'amount', 'salary', 'age', 'price']):
            data_requirements.append('numeric_data')
        if any(word in query_lower for word in ['category', 'department', 'type', 'class']):
            data_requirements.append('categorical_data')
        
        return {
            'intents': detected_intents,
            'complexity': complexity,
            'data_requirements': data_requirements,
            'primary_intent': detected_intents[0] if detected_intents else 'general'
        }
    
    def generate_enhanced_prompt(self, query: str, context: str, dataset_info: Dict[str, Any], 
                                query_type: str = None) -> str:
        """Generate an enhanced prompt based on query analysis"""
        
        # Analyze query intent
        intent_analysis = self.analyze_query_intent(query)
        primary_intent = query_type or intent_analysis['primary_intent']
        
        # Get appropriate template
        template = self.prompt_templates.get(primary_intent, self.prompt_templates['general'])
        
        # Optimize context
        optimized_context = self._optimize_context(context, query, intent_analysis)
        
        # Add few-shot examples if available
        few_shot_context = self._add_few_shot_examples(query, primary_intent)
        
        # Format dataset info
        dataset_info_str = self._format_dataset_info(dataset_info)
        
        # Generate the prompt
        prompt = template.format(
            dataset_info=dataset_info_str,
            context=optimized_context,
            query=query,
            columns=', '.join(dataset_info.get('columns', []))
        )
        
        # Add few-shot examples if helpful
        if few_shot_context and intent_analysis['complexity'] == 'complex':
            prompt = few_shot_context + "\n\n" + prompt
        
        return prompt
    
    def _optimize_context(self, context: str, query: str, intent_analysis: Dict[str, Any]) -> str:
        """Optimize context based on query intent and complexity"""
        optimized_context = context
        
        # Apply context optimizers based on intent
        if intent_analysis['complexity'] == 'complex':
            optimized_context = self.context_optimizers['prioritize'](optimized_context, query)
        
        if len(optimized_context) > 2000:
            optimized_context = self.context_optimizers['truncate'](optimized_context)
        
        if intent_analysis['primary_intent'] == 'statistical':
            optimized_context = self.context_optimizers['summarize'](optimized_context)
        
        return optimized_context
    
    def _add_few_shot_examples(self, query: str, query_type: str) -> str:
        """Add few-shot examples if available and relevant"""
        examples = self.few_shot_examples.get(query_type, [])
        if not examples:
            return ""
        
        # Find the most relevant example
        query_terms = set(query.lower().split())
        best_example = None
        best_score = 0
        
        for example in examples:
            example_terms = set(example['question'].lower().split())
            overlap = len(query_terms.intersection(example_terms))
            if overlap > best_score:
                best_score = overlap
                best_example = example
        
        if best_example and best_score > 1:
            return f"""
**Example Question:** {best_example['question']}
**Example Context:** {best_example['context']}
**Example Answer:** {best_example['answer']}
"""
        
        return ""
    
    def _format_dataset_info(self, dataset_info: Dict[str, Any]) -> str:
        """Format dataset information for prompt inclusion"""
        info_parts = []
        
        if 'shape' in dataset_info:
            rows, cols = dataset_info['shape']
            info_parts.append(f"- Dataset size: {rows:,} rows × {cols} columns")
        
        if 'columns' in dataset_info:
            info_parts.append(f"- Columns: {', '.join(dataset_info['columns'])}")
        
        if 'numeric_columns' in dataset_info and 'categorical_columns' in dataset_info:
            num_numeric = len(dataset_info['numeric_columns'])
            num_categorical = len(dataset_info['categorical_columns'])
            info_parts.append(f"- Data types: {num_numeric} numeric, {num_categorical} categorical")
        
        if 'missing_values' in dataset_info:
            total_missing = sum(dataset_info['missing_values'].values())
            if total_missing > 0:
                info_parts.append(f"- Missing values: {total_missing:,} total")
        
        return '\n'.join(info_parts)
    
    def create_system_prompt(self, dataset_info: Dict[str, Any]) -> str:
        """Create a system prompt for the LLM"""
        return f"""
You are an expert data analyst and AI assistant specializing in CSV data analysis. You have access to a dataset with the following characteristics:

{self._format_dataset_info(dataset_info)}

**Your Capabilities:**
- Statistical analysis and calculations
- Data filtering and aggregation
- Data visualization recommendations
- Pattern recognition and insights
- Data quality assessment

**Your Approach:**
1. Always base your answers on the provided data context
2. Provide clear, actionable insights
3. Include relevant statistics and calculations
4. Suggest follow-up analyses when appropriate
5. Be precise and avoid speculation

**Response Format:**
- Start with a direct answer to the question
- Provide supporting evidence from the data
- Include relevant statistics or calculations
- Suggest additional insights or analyses
- End with actionable recommendations if applicable

Please provide comprehensive, accurate, and insightful responses based on the available data.
"""

def create_enhanced_prompt_engineer() -> AdvancedPromptEngineer:
    """Create an enhanced prompt engineer instance"""
    return AdvancedPromptEngineer()
