"""DSPy module for meme generation."""

from typing import List, Optional, Dict, Any
import dspy
from ..config.config import settings
import random


def ensure_dspy_configured() -> None:
    """Ensure DSPy is configured with a language model."""
    try:
        # Check if DSPy is already configured
        _ = dspy.settings.lm
    except AttributeError:
        # Configure DSPy with LM using the correct API
        lm = dspy.LM(
            f"openai/{settings.dspy_model_name}",
            api_key=settings.openai_api_key
        )
        dspy.configure(lm=lm)


class MemeSignature(dspy.Signature):
    """Signature for meme generation."""
    
    topic: str = dspy.InputField(desc="The topic or theme for the meme")
    format: str = dspy.InputField(desc="The meme format to use (e.g., 'standard', 'modern', 'comparison')")
    context: Optional[str] = dspy.InputField(desc="Additional context or requirements for the meme")
    
    text: str = dspy.OutputField(desc="The text content for the meme")
    image_prompt: str = dspy.OutputField(desc="A detailed prompt for image generation")
    rationale: str = dspy.OutputField(desc="Explanation of why this meme would be effective")
    score: float = dspy.OutputField(desc="A score between 0 and 1 indicating the predicted effectiveness")


class MemePredictor:
    """
    Meme text prediction using DSPy.
    
    In this implementation, we provide a fallback that doesn't require
    DSPy to be configured, for demonstration purposes.
    """
    
    def __init__(self):
        """Initialize the meme predictor."""
        self.funny_texts = [
            "When you finally fix a bug after hours of debugging",
            "Me explaining my code to my rubber duck",
            "When someone asks if my code is production ready",
            "My code: works on my machine",
            "Programmers when they see spaghetti code",
            "When the code works but you don't know why",
        ]
    
    def forward(self, topic: str, format: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate meme text based on topic and format.
        
        Args:
            topic: The topic for the meme
            format: The meme format to use
            context: Optional additional context
            
        Returns:
            Dict containing text, image_prompt and score
        """
        base_text = random.choice(self.funny_texts)
        text = f"{base_text} - {topic}"
        
        # Generate image prompt
        image_prompt = f"A funny meme about {topic} showing {base_text.lower()}, digital art, {format} style"
        
        # Calculate a fake "quality" score
        score = round(random.uniform(0.85, 0.98), 2)
        
        return {
            "text": text,
            "image_prompt": image_prompt,
            "score": score
        }


class TrendPredictor(dspy.Module):
    """DSPy module for predicting meme trends."""
    
    def __init__(self) -> None:
        """Initialize the TrendPredictor."""
        super().__init__()
        ensure_dspy_configured()
        self.predict_trends = dspy.ChainOfThought(dspy.Signature({
            "current_trends": dspy.InputField(desc="List of current trending topics"),
            "predicted_trends": dspy.OutputField(desc="List of predicted upcoming meme trends"),
            "rationale": dspy.OutputField(desc="Explanation for the trend predictions")
        }))
    
    def forward(self, current_trends: List[str]) -> Dict[str, Any]:
        """
        Predict upcoming meme trends based on current trends.
        
        Args:
            current_trends: List of current trending topics
            
        Returns:
            Dictionary containing predicted trends and rationale
        """
        result = self.predict_trends(current_trends=", ".join(current_trends))
        
        return {
            "predicted_trends": [
                trend.strip()
                for trend in result.predicted_trends.split(",")
            ],
            "rationale": result.rationale
        } 