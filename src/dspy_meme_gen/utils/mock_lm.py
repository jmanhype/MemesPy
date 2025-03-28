"""Mock language model for testing without API keys."""

import dspy
import logging
import json
import random

logger = logging.getLogger(__name__)


class MockLM(dspy.LM):
    """Mock language model for testing DSPy without API keys."""
    
    def __init__(self, model_name="mock-lm"):
        """Initialize the mock LM."""
        super().__init__(model=model_name)
        logger.info(f"Initialized MockLM with model name: {model_name}")
        
        # Define some default responses for different types of queries
        self.meme_texts = {
            "Python": [
                "When your code works on the first try: *happy snake noises*",
                "Me debugging for hours only to find I missed a colon",
                "Python: Where indentation is not just a style choice, it's the law"
            ],
            "Machine Learning": [
                "When the model's accuracy is 99.9% on the training data but 50% on test data",
                "Me explaining overfitting to my boss",
                "Data scientists: 80% cleaning data, 20% complaining about cleaning data"
            ],
            "AI": [
                "When the AI assistant understands your typo-filled request perfectly",
                "ChatGPT writing an essay vs me writing an essay",
                "AI in movies vs AI in reality"
            ],
            "Programming": [
                "The code I wrote vs The code I understand",
                "When you fix one bug and create three more",
                "My code doesn't work and I don't know why. My code works and I don't know why."
            ]
        }
        
        self.trending_topics = [
            {
                "id": "ai-ethics",
                "name": "AI Ethics",
                "description": "Discussions about ethical considerations in AI development",
                "popularity": 0.92,
                "suggested_formats": ["comparison", "standard"]
            },
            {
                "id": "coding-struggles",
                "name": "Coding Struggles",
                "description": "Humorous takes on programming challenges",
                "popularity": 0.88,
                "suggested_formats": ["standard", "modern"]
            },
            {
                "id": "data-science",
                "name": "Data Science",
                "description": "Memes about data analysis and visualization",
                "popularity": 0.85,
                "suggested_formats": ["comparison", "standard"]
            }
        ]
        
        self.meme_formats = [
            {
                "id": "drake",
                "name": "Drake Prefer",
                "description": "Two-panel format showing rejection and preference",
                "popularity": 0.95
            },
            {
                "id": "distracted",
                "name": "Distracted Boyfriend",
                "description": "Three-person format showing shifting attention",
                "popularity": 0.92
            },
            {
                "id": "expanding-brain",
                "name": "Expanding Brain",
                "description": "Multi-panel format showing increasingly complex ideas",
                "popularity": 0.89
            }
        ]
    
    def _get_meme_text(self, topic):
        """Get a mock meme text for a given topic."""
        # Check if we have specific responses for this topic
        for key, texts in self.meme_texts.items():
            if key.lower() in topic.lower():
                return random.choice(texts)
        
        # Default fallback responses
        fallbacks = [
            f"A funny meme about {topic}",
            f"When you're trying to understand {topic}",
            f"{topic}: Expectations vs Reality"
        ]
        return random.choice(fallbacks)
    
    def _format_dspy_response(self, content, signature):
        """Format the response based on the signature."""
        if hasattr(signature, 'output_fields'):
            # Handle multiple output fields
            result = {}
            for field_name in signature.output_fields:
                if field_name == "text" or field_name == "answer":
                    result[field_name] = content
                elif field_name == "trending_topics":
                    result[field_name] = self.trending_topics
                elif field_name == "suggested_formats":
                    result[field_name] = self.meme_formats
                else:
                    result[field_name] = f"Mock response for {field_name}"
            return result
        else:
            # Simple single output
            return content
    
    def basic_request(self, prompt, **kwargs):
        """Process a basic request and return a mock response."""
        logger.debug(f"Mock LM received request: {prompt[:100]}...")
        
        # Extract topic from prompt if possible
        topic = "general"
        if "topic" in prompt.lower():
            lines = prompt.lower().split('\n')
            for line in lines:
                if "topic:" in line:
                    topic_part = line.split("topic:")[1].strip()
                    topic = topic_part.split()[0] if topic_part else "general"
        
        # Generate a mock response
        response_text = self._get_meme_text(topic)
        
        logger.debug(f"Mock LM returning response: {response_text}")
        return {"text": response_text}
    
    def request(self, prompt, **kwargs):
        """Process a request and return a mock response."""
        signature = kwargs.get('signature')
        
        basic_response = self.basic_request(prompt, **kwargs)
        
        if signature:
            return self._format_dspy_response(basic_response["text"], signature)
        else:
            return basic_response


def get_mock_lm():
    """Get a mock LM for testing without an API key."""
    logger.info("Using mock LM since no API key is available")
    return MockLM() 