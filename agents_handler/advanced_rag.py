import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
from datetime import datetime

class AdvancedRAGSystem:
    """
    Advanced RAG system with multiple retrieval strategies and improved context management
    """
    
    def __init__(self, df: pd.DataFrame, embedding_model=None):
        self.df = df
        self.embedding_model = embedding_model
        self.vectorizer = None
        self.tfidf_matrix = None
        self.row_embeddings = None
        self.column_embeddings = None
        self.cache = {}
        self.query_history = []
        
        # Initialize the system
        self._initialize_embeddings()
        self._create_metadata()
    
    def _initialize_embeddings(self):
        """Initialize TF-IDF embeddings for semantic search"""
        try:
            # Create text representations of each row
            row_texts = []
            for idx, row in self.df.iterrows():
                row_text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
                row_texts.append(row_text)
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(row_texts)
            
            # Create column embeddings
            column_texts = [f"column: {col} type: {self.df[col].dtype}" for col in self.df.columns]
            self.column_embeddings = self.vectorizer.transform(column_texts)
            
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            self.vectorizer = None
            self.tfidf_matrix = None
    
    def _create_metadata(self):
        """Create metadata for enhanced retrieval"""
        self.metadata = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_counts': {col: self.df[col].nunique() for col in self.df.columns},
            'numeric_columns': list(self.df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime']).columns)
        }
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[pd.Series, float]]:
        """Semantic search using TF-IDF embeddings"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((self.df.iloc[idx], similarities[idx]))
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[pd.Series, float]]:
        """Enhanced keyword-based search"""
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall',
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those'
        }
        query_terms = query_terms - stop_words
        
        results = []
        for idx, row in self.df.iterrows():
            score = 0
            row_text = " ".join([str(val).lower() for val in row.values])
            
            # Exact term matches
            for term in query_terms:
                if term in row_text:
                    score += 1
                    # Bonus for column name matches
                    for col in self.df.columns:
                        if term in col.lower():
                            score += 0.5
            
            # Phrase matching
            if len(query_terms) > 1:
                phrase = " ".join(query_terms)
                if phrase in row_text:
                    score += 2
            
            # Column-specific matching
            for col in self.df.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in query_terms):
                    score += 0.3
            
            if score > 0:
                results.append((row, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[pd.Series]:
        """Hybrid search combining semantic and keyword search"""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and deduplicate results
        combined_scores = {}
        
        # Add semantic results
        for row, score in semantic_results:
            row_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            combined_scores[row_hash] = {
                'row': row,
                'semantic_score': score,
                'keyword_score': 0,
                'combined_score': score * semantic_weight
            }
        
        # Add keyword results
        for row, score in keyword_results:
            row_hash = hashlib.md5(str(row.values).encode()).hexdigest()
            if row_hash in combined_scores:
                combined_scores[row_hash]['keyword_score'] = score
                combined_scores[row_hash]['combined_score'] += score * (1 - semantic_weight)
            else:
                combined_scores[row_hash] = {
                    'row': row,
                    'semantic_score': 0,
                    'keyword_score': score,
                    'combined_score': score * (1 - semantic_weight)
                }
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return [item['row'] for item in sorted_results[:top_k]]
    
    def context_aware_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Context-aware search with metadata and query analysis"""
        # Analyze query type
        query_type = self._analyze_query_type(query)
        
        # Get relevant rows based on query type
        if query_type == 'statistical':
            relevant_rows = self._statistical_search(query, top_k)
        elif query_type == 'filtering':
            relevant_rows = self._filtering_search(query, top_k)
        elif query_type == 'aggregation':
            relevant_rows = self._aggregation_search(query, top_k)
        else:
            relevant_rows = self.hybrid_search(query, top_k)
        
        # Create enhanced context
        context = self._create_enhanced_context(query, relevant_rows, query_type)
        
        return {
            'relevant_rows': relevant_rows,
            'context': context,
            'query_type': query_type,
            'metadata': self.metadata
        }
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the type of query for better retrieval"""
        query_lower = query.lower()
        
        # Statistical queries
        stat_keywords = ['average', 'mean', 'median', 'sum', 'count', 'total', 'percentage', 'correlation']
        if any(keyword in query_lower for keyword in stat_keywords):
            return 'statistical'
        
        # Filtering queries
        filter_keywords = ['where', 'filter', 'show', 'find', 'select', 'rows where', 'data where']
        if any(keyword in query_lower for keyword in filter_keywords):
            return 'filtering'
        
        # Aggregation queries
        agg_keywords = ['group by', 'grouped', 'by category', 'by department', 'by type']
        if any(keyword in query_lower for keyword in agg_keywords):
            return 'aggregation'
        
        return 'general'
    
    def _statistical_search(self, query: str, top_k: int) -> List[pd.Series]:
        """Search optimized for statistical queries"""
        # For statistical queries, we want a good sample of the data
        if len(self.df) <= top_k:
            return [row for _, row in self.df.iterrows()]
        
        # Stratified sampling for better statistical representation
        numeric_cols = self.metadata['numeric_columns']
        if numeric_cols:
            # Sample from different ranges of numeric data
            sample_size = min(top_k, len(self.df))
            return self.df.sample(n=sample_size, random_state=42).to_dict('records')
        else:
            return self.df.head(top_k).to_dict('records')
    
    def _filtering_search(self, query: str, top_k: int) -> List[pd.Series]:
        """Search optimized for filtering queries"""
        # Use hybrid search but prioritize exact matches
        return self.hybrid_search(query, top_k, semantic_weight=0.3)
    
    def _aggregation_search(self, query: str, top_k: int) -> List[pd.Series]:
        """Search optimized for aggregation queries"""
        # For aggregation, we need representative samples from different groups
        categorical_cols = self.metadata['categorical_columns']
        if categorical_cols:
            # Sample from different categories
            samples = []
            for col in categorical_cols[:3]:  # Use first 3 categorical columns
                unique_values = self.df[col].unique()[:5]  # First 5 unique values
                for value in unique_values:
                    subset = self.df[self.df[col] == value].head(2)
                    samples.extend(subset.to_dict('records'))
                    if len(samples) >= top_k:
                        break
                if len(samples) >= top_k:
                    break
            return samples[:top_k]
        else:
            return self.df.head(top_k).to_dict('records')
    
    def _create_enhanced_context(self, query: str, relevant_rows: List[pd.Series], query_type: str) -> str:
        """Create enhanced context with metadata and query-specific information"""
        context_parts = []
        
        # Add dataset overview
        context_parts.append(f"Dataset Overview:")
        context_parts.append(f"- Shape: {self.metadata['shape'][0]:,} rows × {self.metadata['shape'][1]} columns")
        context_parts.append(f"- Columns: {', '.join(self.metadata['columns'])}")
        context_parts.append(f"- Data Types: {len(self.metadata['numeric_columns'])} numeric, {len(self.metadata['categorical_columns'])} categorical")
        
        # Add query-specific context
        if query_type == 'statistical':
            context_parts.append(f"- Numeric columns available: {', '.join(self.metadata['numeric_columns'])}")
        elif query_type == 'filtering':
            context_parts.append(f"- Categorical columns for filtering: {', '.join(self.metadata['categorical_columns'])}")
        
        # Add relevant rows
        context_parts.append(f"\nRelevant Data Rows:")
        for i, row in enumerate(relevant_rows, 1):
            row_text = " | ".join([f"{col}: {row[col]}" for col in self.df.columns])
            context_parts.append(f"Row {i}: {row_text}")
        
        return "\n".join(context_parts)
    
    def cache_query(self, query: str, results: Dict[str, Any]):
        """Cache query results for faster retrieval"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = {
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'query': query
        }
        
        # Keep only recent queries in cache
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def get_cached_results(self, query: str) -> Dict[str, Any]:
        """Get cached results if available"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.cache.get(query_hash, None)

def create_advanced_rag_agent(df: pd.DataFrame, llm, top_k: int = 5):
    """
    Create an advanced RAG agent with multiple retrieval strategies
    """
    rag_system = AdvancedRAGSystem(df)
    
    def advanced_rag_query(user_query: str) -> str:
        # Check cache first
        cached_results = rag_system.get_cached_results(user_query)
        if cached_results:
            relevant_rows = cached_results['results']['relevant_rows']
            context = cached_results['results']['context']
        else:
            # Perform context-aware search
            search_results = rag_system.context_aware_search(user_query, top_k)
            relevant_rows = search_results['relevant_rows']
            context = search_results['context']
            
            # Cache results
            rag_system.cache_query(user_query, search_results)
        
        # Create enhanced prompt
        enhanced_prompt = f"""
You are an advanced data assistant using Retrieval-Augmented Generation (RAG) with multiple search strategies.

{context}

**User Question:** {user_query}

**Instructions:**
1. Use the provided context to answer the question accurately
2. If the answer requires calculations, show your reasoning
3. If the answer is not in the data, clearly state what information is missing
4. Provide specific, actionable insights when possible
5. Use the dataset metadata to provide context about data quality

**Available Columns:** {', '.join(df.columns)}

Please provide a comprehensive, accurate answer based on the retrieved data and dataset context.
"""
        
        # Get response from LLM
        response = llm.invoke(enhanced_prompt)
        
        # Clean response
        from ui_components.response_cleaner import clean_llm_response
        return clean_llm_response(response)
    
    return advanced_rag_query
