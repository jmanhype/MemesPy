#!/usr/bin/env python3
"""
Direct test of OpenAI API using the official Python client.
"""
import re
import os
from openai import OpenAI, APIError

def read_api_key_from_env_file() -> str:
    """
    Read the OpenAI API key directly from the .env file
    
    Returns:
        str: The OpenAI API key
    """
    try:
        with open('.env', 'r') as f:
            content = f.read()
            
        # Use regex to find the OPENAI_API_KEY line
        match = re.search(r'OPENAI_API_KEY=([^\n]+)', content)
        if match:
            return match.group(1).strip()
        else:
            print("Could not find OPENAI_API_KEY in .env file")
            return ""
    except Exception as e:
        print(f"Error reading .env file: {e}")
        return ""

def main():
    """Test the OpenAI API directly with the official client."""
    api_key = read_api_key_from_env_file()
    if not api_key:
        print("API key not found")
        return
    
    print(f"Testing API key (first 10 chars): {api_key[:10]}...")
    
    # Test 1: Basic client with just API key
    print("\nTest 1: Basic client with just API key")
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, who are you?"}
            ],
            max_tokens=50
        )
        print("Success! API key is valid.")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: With project_id parameter
    if api_key.startswith("sk-proj-"):
        # Extract the project ID part
        parts = api_key.split("-")
        if len(parts) >= 3:
            project_id = parts[2]
            
            print(f"\nTest 2: With project_id parameter (extracted: {project_id[:8]}...)")
            try:
                client = OpenAI(
                    api_key=api_key, 
                    project=project_id  # Try using project ID parameter
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Tell me about Python programming briefly."}
                    ],
                    max_tokens=50
                )
                print("Success! API key with project_id is valid.")
                print(f"Response: {response.choices[0].message.content}")
            except Exception as e:
                print(f"Error: {e}")
    
    # Test 3: Try with an empty organization
    print("\nTest 3: With empty organization")
    try:
        client = OpenAI(
            api_key=api_key,
            organization=""  # Explicitly set to empty string
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's a meme?"}
            ],
            max_tokens=50
        )
        print("Success! API key with empty organization is valid.")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Try with default client (using environment variable)
    print("\nTest 4: With environment variable (os.environ)")
    try:
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Create client without explicit api_key parameter
        client = OpenAI()  # This should pick up the environment variable
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a short funny meme about Python programming."}
            ],
            max_tokens=100
        )
        print("Success! Environment variable method works.")
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 