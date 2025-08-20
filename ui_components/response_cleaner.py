import re

def clean_llm_response(response):
    """
    Comprehensive function to clean LLM responses and remove all metadata.
    Handles different response formats from various LLM providers.
    """
    if response is None:
        return ""
    
    # Extract content from different response object types
    if hasattr(response, 'content'):
        clean_response = response.content
    elif hasattr(response, 'text'):
        clean_response = response.text
    elif hasattr(response, 'response'):
        clean_response = response.response
    elif hasattr(response, 'message') and hasattr(response.message, 'content'):
        clean_response = response.message.content
    elif isinstance(response, str):
        clean_response = response
    else:
        clean_response = str(response)
    
    # If it's a string with metadata, try to extract just the content
    if isinstance(clean_response, str):
        # Try to extract content from various patterns
        content_patterns = [
            r"content='([^']*)'",
            r'content="([^"]*)"',
            r"content=([^,\s}]*)",
            r"text='([^']*)'",
            r'text="([^"]*)"',
            r"text=([^,\s}]*)"
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, clean_response)
            if match:
                clean_response = match.group(1)
                break
    
    # Comprehensive metadata removal
    metadata_patterns = [
        r'additional_kwargs=\{.*?\}',
        r'response_metadata=\{.*?\}',
        r'usage_metadata=\{.*?\}',
        r'id=\'[^\']*\'',
        r'finish_reason=\'[^\']*\'',
        r'model_name=\'[^\']*\'',
        r'system_fingerprint=\'[^\']*\'',
        r'service_tier=\'[^\']*\'',
        r'logprobs=\'[^\']*\'',
        r'Available columns:.*',
        r'content=\'',
        r'content="',
        r'text=\'',
        r'text="',
        r'generation_info=\{.*?\}',
        r'prompt_feedback=\{.*?\}',
        r'candidates=\[.*?\]',
        r'usage_metadata=\{.*?\}',
        r'token_usage=\{.*?\}',
        r'completion_tokens=\d+',
        r'prompt_tokens=\d+',
        r'total_tokens=\d+',
        r'completion_time=[\d.]+',
        r'prompt_time=[\d.]+',
        r'queue_time=[\d.]+',
        r'total_time=[\d.]+',
        r'run_id=\'[^\']*\'',
        r'run_type=\'[^\']*\'',
        r'start_time=[\d.]+',
        r'end_time=[\d.]+',
        r'extra=\{[^}]*\}',
        r'name=\'[^\']*\'',
        r'type=\'[^\']*\'',
        r'data=\{[^}]*\}',
        r'is_chunk=\w+',
        r'generation=\{[^}]*\}',
        r'usage=\{[^}]*\}',
        r'prompt_tokens=\d+',
        r'completion_tokens=\d+',
        r'total_tokens=\d+'
    ]
    
    for pattern in metadata_patterns:
        clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining curly braces and their content
    clean_response = re.sub(r'\{[^}]*\}', '', clean_response)
    
    # Remove any remaining square brackets and their content
    clean_response = re.sub(r'\[[^\]]*\]', '', clean_response)
    
    # Remove any remaining parentheses with metadata-like content
    clean_response = re.sub(r'\([^)]*metadata[^)]*\)', '', clean_response, flags=re.IGNORECASE)
    
    # Clean up extra whitespace, newlines, and punctuation
    clean_response = re.sub(r'\s+', ' ', clean_response)
    clean_response = re.sub(r'^\s*[\'"]\s*', '', clean_response)  # Remove leading quotes
    clean_response = re.sub(r'\s*[\'"]\s*$', '', clean_response)  # Remove trailing quotes
    clean_response = re.sub(r'^\s*,\s*', '', clean_response)  # Remove leading commas
    clean_response = re.sub(r'\s*,\s*$', '', clean_response)  # Remove trailing commas
    
    return clean_response.strip()
