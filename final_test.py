"""Test DSPy using a mock language model."""

import dspy
import json
import uuid
from datetime import datetime

# Create a custom DSPy language model that just returns predefined responses
class MockLanguageModel(dspy.LM):
    """Mock language model for testing."""
    
    def __init__(self):
        """Initialize the mock LM."""
        super().__init__(model="mock")
    
    def basic_request(self, prompt, **kwargs):
        """Return a predefined response based on the prompt."""
        print(f"Received prompt: {prompt}")
        
        # Check if the prompt contains certain keywords to determine response
        if "meme" in prompt.lower():
            return json.dumps({
                "meme_text": "Python: Where indentation is not just a style choice, it's a lifestyle."
            })
        elif "trend" in prompt.lower():
            return json.dumps({
                "trending_topics": [
                    {
                        "id": "python-jokes",
                        "name": "Python Jokes",
                        "description": "Humorous takes on Python programming",
                        "popularity": 0.9,
                        "suggested_formats": ["standard", "modern"]
                    },
                    {
                        "id": "ai-ethics",
                        "name": "AI Ethics",
                        "description": "Discussions about ethical considerations in AI",
                        "popularity": 0.85,
                        "suggested_formats": ["comparison", "modern"]
                    }
                ]
            })
        elif "format" in prompt.lower():
            return json.dumps({
                "formats": [
                    {
                        "id": "standard",
                        "name": "Standard",
                        "description": "Classic top/bottom text meme format",
                        "popularity": 0.9
                    },
                    {
                        "id": "modern",
                        "name": "Modern",
                        "description": "Contemporary meme format with integrated text",
                        "popularity": 0.8
                    },
                    {
                        "id": "comparison",
                        "name": "Comparison",
                        "description": "Side-by-side comparison format",
                        "popularity": 0.7
                    }
                ]
            })
        else:
            return json.dumps({
                "text": "I don't have a specific response for this prompt."
            })


# Define our signatures
class MemePrompt(dspy.Signature):
    """Signature for generating meme text."""
    
    topic: str = dspy.InputField(desc="Topic for the meme")
    format_id: str = dspy.InputField(desc="Format ID for the meme")
    
    meme_text: str = dspy.OutputField(desc="Generated meme text")


class TrendQuery(dspy.Signature):
    """Signature for querying trending topics."""
    
    query: str = dspy.InputField(desc="Query for trends to analyze")
    
    trending_topics: list = dspy.OutputField(desc="List of trending topics with details")


class FormatQuery(dspy.Signature):
    """Signature for querying meme formats."""
    
    query: str = dspy.InputField(desc="Query for meme formats")
    
    formats: list = dspy.OutputField(desc="List of meme formats with details")


# Set up the mock language model
lm = MockLanguageModel()
dspy.settings.configure(lm=lm)

# Create DSPy modules
meme_generator = dspy.Predict(MemePrompt)
trend_analyzer = dspy.Predict(TrendQuery)
format_generator = dspy.Predict(FormatQuery)


# Test meme generation
def test_meme_generation():
    """Test meme generation."""
    topic = "Python Programming"
    format_id = "standard"
    
    print(f"\nGenerating meme for topic '{topic}' with format '{format_id}'...")
    prediction = meme_generator(topic=topic, format_id=format_id)
    
    meme = {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "format": format_id,
        "text": prediction.meme_text,
        "image_url": "https://example.com/placeholder.jpg",
        "created_at": datetime.now().isoformat(),
        "score": 0.8
    }
    
    print(f"Generated meme: {meme['text']}")
    return meme


# Test trend analysis
def test_trend_analysis():
    """Test trend analysis."""
    query = "current internet meme trends"
    
    print(f"\nAnalyzing trends with query: '{query}'...")
    prediction = trend_analyzer(query=query)
    
    print(f"Found {len(prediction.trending_topics)} trending topics:")
    for topic in prediction.trending_topics:
        print(f"  - {topic['name']}: {topic['description']}")
    
    return prediction.trending_topics


# Test format generation
def test_format_generation():
    """Test format generation."""
    query = "popular meme formats"
    
    print(f"\nGenerating formats with query: '{query}'...")
    prediction = format_generator(query=query)
    
    print(f"Generated {len(prediction.formats)} meme formats:")
    for format_item in prediction.formats:
        print(f"  - {format_item['name']}: {format_item['description']}")
    
    return prediction.formats


if __name__ == "__main__":
    print("Testing DSPy Meme Generator with mock language model...")
    
    # Run all tests
    meme = test_meme_generation()
    topics = test_trend_analysis()
    formats = test_format_generation()
    
    print("\nAll tests completed successfully!") 