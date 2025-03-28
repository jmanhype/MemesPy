"""Simple FastAPI server for testing."""
from typing import Dict, Any
import os
import dspy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Simple DSPy Test API",
    description="A simple API for testing DSPy with curl",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {"message": "Hello, World!"}

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    # Check if OpenAI API key is configured
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_configured = bool(openai_api_key)
    
    # Get the first few characters of the API key for debugging
    api_key_prefix = openai_api_key[:8] + "..." if openai_api_key else "not set"
    
    # Basic health check response
    return {
        "status": "healthy",
        "version": "0.1.0",
        "env": os.environ.get("ENVIRONMENT", "development"),
        "openai": {
            "api_key_prefix": api_key_prefix,
            "openai_configured": openai_configured
        }
    }

@app.get("/api/generate-meme")
async def generate_meme(topic: str = "Python Programming") -> Dict[str, Any]:
    """Generate meme endpoint."""
    try:
        # Configure DSPy
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return {"error": "OpenAI API key not set"}
            
        # Set up DSPy
        lm = dspy.LM(
            model="openai/gpt-3.5-turbo-0125",
            api_key=api_key
        )
        dspy.settings.configure(lm=lm)
        
        # Define the MemePrompt
        class MemePrompt(dspy.Signature):
            """Generate a meme about a given topic."""
            topic = dspy.InputField(desc="The topic to generate a meme about")
            text = dspy.OutputField(desc="The meme text")
        
        # Create the generator
        generator = dspy.Predict(MemePrompt)
        
        # Generate the meme
        result = generator(topic=topic)
        
        # Return the result
        return {
            "topic": topic,
            "text": result.text,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 