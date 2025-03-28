#!/usr/bin/env python3
"""
Direct test of OpenAI API to validate the API key.
"""
import os
import re
import json
import requests

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
    """
    Test the OpenAI API directly without using DSPy.
    """
    api_key = read_api_key_from_env_file()
    if not api_key:
        print("API key not found")
        return

    print(f"Testing API key (first 10 chars): {api_key[:10]}...")

    # Define the API endpoint for a simple models list request
    url = "https://api.openai.com/v1/models"
    
    # First attempt: without organization header
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    print("\nTest 1: Without organization header")
    try:
        response = requests.get(url, headers=headers)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Success! API key is valid.")
            # Print the first 5 models
            models = response.json().get("data", [])
            print(f"Found {len(models)} models. First 5:")
            for model in models[:5]:
                print(f"- {model.get('id')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error making request: {e}")
    
    # Second attempt: try using a derived organization ID
    if api_key.startswith("sk-proj-"):
        # Extract possible project ID from the key
        parts = api_key.split("-")
        if len(parts) >= 3:
            org_id = parts[2]
            
            print(f"\nTest 2: With organization header (extracted: {org_id[:8]}...)")
            headers["OpenAI-Organization"] = org_id
            
            try:
                response = requests.get(url, headers=headers)
                print(f"Status code: {response.status_code}")
                if response.status_code == 200:
                    print("Success! API key and organization ID are valid.")
                    # Print the first 5 models
                    models = response.json().get("data", [])
                    print(f"Found {len(models)} models. First 5:")
                    for model in models[:5]:
                        print(f"- {model.get('id')}")
                else:
                    print(f"Error response: {response.text}")
            except Exception as e:
                print(f"Error making request: {e}")
    
    # Third attempt: Try using the first portion of the API key
    if api_key.startswith("sk-proj-"):
        # Get the portion after "sk-proj-" prefix
        prefix = api_key[8:]
        
        # Try using the first section before any hyphens
        org_id = prefix.split("-")[0] if "-" in prefix else prefix
        
        print(f"\nTest 3: With organization header (first segment: {org_id[:8]}...)")
        headers["OpenAI-Organization"] = org_id
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                print("Success! API key and organization ID are valid.")
                # Print the first 5 models
                models = response.json().get("data", [])
                print(f"Found {len(models)} models. First 5:")
                for model in models[:5]:
                    print(f"- {model.get('id')}")
            else:
                print(f"Error response: {response.text}")
        except Exception as e:
            print(f"Error making request: {e}")

    # Fourth attempt: Without any organization header, but using v1beta endpoint
    url_beta = "https://api.openai.com/v1beta/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    print("\nTest 4: Using v1beta endpoint without organization header")
    try:
        response = requests.get(url_beta, headers=headers)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Success! API key is valid for beta endpoint.")
            # Print the first 5 models
            models = response.json().get("data", [])
            print(f"Found {len(models)} models. First 5:")
            for model in models[:5]:
                print(f"- {model.get('id')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    print("Testing OpenAI API directly...")
    
    # Set environment variables explicitly
    print("Setting environment variables...")
    api_key = read_api_key_from_env_file()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"Set OPENAI_API_KEY environment variable (first 10 chars): {api_key[:10]}...")
        
        # Set empty organization
        os.environ["OPENAI_ORG_ID"] = ""
        print("Set OPENAI_ORG_ID to empty string")
    
    # Run the test
    main()
    print("Test completed.")
    
    # Print a command that can be used to run DSPy with these environment variables
    print("\nTo run DSPy with these environment variables, use:")
    print("OPENAI_API_KEY=\"" + api_key + "\" OPENAI_ORG_ID=\"\" PYTHONPATH=. python dspy_minimal_test.py") 