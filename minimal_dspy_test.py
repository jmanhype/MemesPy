"""Minimal test script for DSPy."""

import os
import dspy
import asyncio
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime


class MemePrompt(dspy.Signature):
    """Signature for the meme generation prompt."""
    
    topic: str = dspy.InputField(desc="The topic for the meme")
    format: str = dspy.InputField(desc="The meme format to use")
    style: Optional[str] = dspy.InputField(desc="Optional style for the meme")
    
    text: str = dspy.OutputField(desc="The generated meme text")


async def main():
    """Test DSPy implementation."""
    # Set up the DSPy language model
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return
    
    print(f"Using OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-5:]}")
    
    # Create LM instance
    lm = dspy.LM(
        model=f"openai/gpt-3.5-turbo-0125",
        api_key=openai_api_key
    )
    
    # Set the LM as the default
    dspy.settings.configure(lm=lm)
    
    # Create the meme generator module
    generator = dspy.Predict(MemePrompt)
    
    # Test topics
    topics = ["Python Programming", "Machine Learning", "DSPy"]
    
    for topic in topics:
        print(f"\nGenerating meme for topic: {topic}")
        
        # Generate the meme text using DSPy
        prediction = generator(
            topic=topic,
            format="standard",
            style="funny"
        )
        
        # Create a response with the generated text
        meme_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        result = {
            "id": meme_id,
            "topic": topic,
            "format": "standard",
            "text": prediction.text,
            "image_url": f"https://example.com/memes/{meme_id}.jpg",
            "created_at": created_at,
            "score": 0.9
        }
        
        print(f"Generated meme text: {prediction.text}")


if __name__ == "__main__":
    asyncio.run(main()) 