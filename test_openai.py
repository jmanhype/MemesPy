"""Simple script to test OpenAI API directly."""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
print("Loading .env file...")
load_dotenv(verbose=True)

# Debug: Read the API key directly from the file
print("Reading API key directly from .env file...")
with open('.env', 'r') as f:
    env_contents = f.read()
    for line in env_contents.split('\n'):
        if line.startswith('OPENAI_API_KEY='):
            direct_key = line.strip().split('=', 1)[1]
            print(f"Direct key from file: {direct_key}")
            break

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
print(f"API Key from environment: {api_key}")

# Compare keys
if direct_key == api_key:
    print("Keys match!")
else:
    print("Keys don't match!")

# Use the direct key for the client
client = OpenAI(
    api_key=direct_key
)

def generate_meme_text(topic: str, format_id: str) -> str:
    """Generate meme text using OpenAI API directly."""
    try:
        print(f"Generating meme for topic: {topic}, format: {format_id}")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates funny meme text."},
                {"role": "user", "content": f"Create a funny meme text about {topic} in the {format_id} format. Keep it short and witty."}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        meme_text = response.choices[0].message.content
        print(f"Generated text: {meme_text}")
        return meme_text
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"


def main():
    """Main function to test OpenAI API."""
    test_topics = [
        "Python Programming"
    ]
    
    for topic in test_topics:
        text = generate_meme_text(topic, "standard")
        print(f"\n--- Meme ---")
        print(f"Topic: {topic}")
        print(f"Text: {text}")


if __name__ == "__main__":
    main() 