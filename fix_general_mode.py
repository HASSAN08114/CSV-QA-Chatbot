#!/usr/bin/env python3
"""
Script to fix the general mode response cleaning in chat_analysis.py
"""

def fix_general_mode():
    # Read the file
    with open('ui_components/chat_analysis.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the general mode cleaning section
    old_section = '''                        # Extract clean content directly from the response object
                        if hasattr(response, 'content'):
                            clean_response = response.content
                        elif hasattr(response, 'text'):
                            clean_response = response.text
                        elif hasattr(response, 'response'):
                            clean_response = response.response
                        elif isinstance(response, str):
                            # If it's already a string, try to extract just the content part
                            if "content='" in response:
                                # More robust content extraction
                                content_match = re.search(r"content='([^']*)'", response)
                                if content_match:
                                    clean_response = content_match.group(1)
                                else:
                                    # Try alternative pattern for content extraction
                                    content_match = re.search(r"content=\\"([^\\"]*)\\"", response)
                                    if content_match:
                                        clean_response = content_match.group(1)
                                    else:
                                        clean_response = response
                            else:
                                clean_response = response
                        else:
                            clean_response = str(response)

                        # Final cleanup - remove any remaining metadata
                        # Remove all metadata patterns more thoroughly
                        clean_response = re.sub(r'additional_kwargs=\\{.*?\\}', '', clean_response, flags=re.DOTALL)
                        clean_response = re.sub(r'response_metadata=\\{.*?\\}', '', clean_response, flags=re.DOTALL)
                        clean_response = re.sub(r'usage_metadata=\\{.*?\\}', '', clean_response, flags=re.DOTALL)
                        clean_response = re.sub(r'id=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'finish_reason=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'model_name=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'system_fingerprint=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'service_tier=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'logprobs=\\'[^\\']*\\'', '', clean_response)
                        clean_response = re.sub(r'Available columns:.*', '', clean_response)
                        clean_response = re.sub(r'content=\\'', '', clean_response)
                        clean_response = re.sub(r'content=\\"', '', clean_response)
                        clean_response = re.sub(r'\\'$', '', clean_response)
                        clean_response = re.sub(r'\\"$', '', clean_response)
                        # Remove any remaining curly braces and their content
                        clean_response = re.sub(r'\\{[^}]*\\}', '', clean_response)
                        # Clean up extra whitespace and newlines
                        clean_response = re.sub(r'\\s+', ' ', clean_response)
                        clean_response = clean_response.strip()'''
    
    new_section = '''                        # Use the existing comprehensive response cleaning function
                        clean_response = clean_llm_response(response)'''
    
    # Replace the section
    new_content = content.replace(old_section, new_section)
    
    # Write back to file
    with open('ui_components/chat_analysis.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully fixed general mode response cleaning in chat_analysis.py")

if __name__ == "__main__":
    fix_general_mode()
