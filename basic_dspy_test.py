"""Basic DSPy test."""

import os
import dspy
import random


class SimpleQA(dspy.Signature):
    """Simple QA signature."""
    
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Answer to the question")


def get_mock_lm():
    """Get a mock LM for testing without an API key."""
    print("Using mock LM since no OpenAI API key is available")
    
    # Define a mock completion function that returns canned responses
    def mock_completion(prompt, **kwargs):
        responses = {
            "Python": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            "DSPy": "DSPy is a framework for programming—rather than prompting—language models, allowing you to build sophisticated AI systems.",
            "meme": "Meme generation typically involves combining images with text that follows specific formats, often leveraging AI for creative content.",
        }
        
        # Find a matching canned response or return a generic one
        for key, response in responses.items():
            if key.lower() in prompt.lower():
                return {"text": response}
        
        return {"text": "I don't have specific information about that topic."}
    
    # Create a simple mock LM
    class MockLM(dspy.LM):
        def __init__(self):
            super().__init__(model="mock")
            
        def basic_request(self, prompt, **kwargs):
            return mock_completion(prompt, **kwargs)
    
    return MockLM()


def main():
    """Test basic DSPy functionality."""
    # Get the API key from the environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key or openai_api_key == "your-openai-api-key-here":
        print("WARNING: Valid OPENAI_API_KEY not set. Using mock responses.")
        lm = get_mock_lm()
    else:
        # Mask the key for logging
        masked_key = openai_api_key[:4] + "..." + openai_api_key[-4:]
        print(f"Using OpenAI API key: {masked_key}")
        
        # Configure DSPy with the API key
        lm = dspy.LM("openai/gpt-3.5-turbo-0125", api_key=openai_api_key)
    
    # Configure DSPy with the LM
    dspy.settings.configure(lm=lm)
    
    # Create a predictor
    predictor = dspy.Predict(SimpleQA)
    
    # Test questions
    questions = [
        "What is Python?",
        "What is DSPy?",
        "How does meme generation work?"
    ]
    
    # Test the predictor
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = predictor(question=question)
            print(f"Answer: {result.answer}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main() 