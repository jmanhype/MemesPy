"""DSPy module for meme generation."""

from typing import List, Optional, Dict, Any, Tuple, Union
import dspy
from ..config.config import settings
import random
import logging
from dspy.signatures import Signature

# Configure logger
logger = logging.getLogger(__name__)

def ensure_dspy_configured() -> bool:
    """Ensure DSPy is configured with a language model.
    
    Returns:
        bool: True if DSPy is configured, False otherwise.
    """
    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured. Cannot configure DSPy.")
        return False
        
    try:
        # Check if DSPy is already configured
        _ = dspy.settings.lm
        logger.info("DSPy already configured.")
        return True
    except AttributeError:
        try:
            # Configure DSPy with LM using the correct API
            logger.info(f"Configuring DSPy with model: {settings.dspy_model}")
            lm = dspy.LM(
                f"openai/{settings.dspy_model}",
                api_key=settings.openai_api_key
            )
            dspy.configure(lm=lm)
            logger.info("DSPy configured successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            return False


class MemeSignature(dspy.Signature):
    """Signature for meme generation."""
    
    topic: str = dspy.InputField(desc="The topic or theme for the meme")
    format: str = dspy.InputField(desc="The meme format to use (e.g., 'standard', 'modern', 'comparison')")
    context: Optional[str] = dspy.InputField(desc="Additional context or requirements for the meme")
    
    text: str = dspy.OutputField(desc="The text content for the meme")
    image_prompt: str = dspy.OutputField(desc="A detailed prompt for image generation")
    rationale: str = dspy.OutputField(desc="Explanation of why this meme would be effective")
    score: float = dspy.OutputField(desc="A score between 0 and 1 indicating the predicted effectiveness")


# Define the DSPy Signature for meme content generation
class GenerateMemeContent(Signature):
    """Generates meme text and a corresponding image prompt based on a topic and format."""

    meme_topic: str = dspy.InputField(desc="The central theme or subject of the meme.")
    meme_format: str = dspy.InputField(desc="The specific meme template or style (e.g., 'drake', 'distracted boyfriend').")
    # context: Optional[str] = dspy.InputField(desc="Optional additional context or instructions.") # Removed context for now

    meme_text: str = dspy.OutputField(desc="The witty text overlay for the meme.")
    image_prompt: str = dspy.OutputField(
        desc="A detailed prompt suitable for an image generation model, describing the visual elements of the meme based on the topic and format."
    )
    # score: float = dspy.OutputField(desc="Confidence score (0.0-1.0) - Removed for simplicity")


class MemePredictor(dspy.Module):
    """DSPy module to predict meme text and image prompt."""

    def __init__(self) -> None:
        """
        Initialize the MemePredictor module.
        
        The module assumes DSPy is already configured externally.
        """
        super().__init__()
        # Initialize predictor assuming DSPy configuration happens externally
        self.generate_meme_content = dspy.Predict(GenerateMemeContent)
        
        # Define fallback texts for when DSPy generation fails
        self.fallback_funny_texts = [
            "When you finally fix a bug after hours of debugging",
            "Me explaining my code to my rubber duck",
            "When someone asks if my code is production ready",
            "My code: works on my machine",
            "Programmers when they see spaghetti code",
            "When the code works but you don't know why",
        ]

    def forward(
        self, topic: str, format: str
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Generate meme text and image prompt using the configured LM or fallback.

        Args:
            topic (str): The topic for the meme.
            format (str): The desired meme format (e.g., 'drake', 'success kid').

        Returns:
            Tuple[Union[str, None], Union[str, None]]: A tuple containing the generated
                                           text and image prompt, or
                                           (None, None) if generation fails.
        """
        text: Optional[str] = None
        image_prompt: Optional[str] = None

        # Check if DSPy LM is configured globally before trying to use it
        try:
            # Try to use DSPy to generate meme content
            logger.info(f"Generating meme content via DSPy for topic: {topic}")
            # Generate meme content using the DSPy predictor
            prediction = self.generate_meme_content(
                meme_topic=topic, meme_format=format
            )
            # Log the raw prediction for debugging
            logger.debug(f"DSPy prediction raw output: {prediction}")

            # Validate prediction fields
            if hasattr(prediction, "meme_text") and hasattr(
                prediction, "image_prompt"
            ):
                text = prediction.meme_text
                image_prompt = prediction.image_prompt
                logger.info(f"DSPy generated text: {text}")
                logger.info(f"DSPy generated image_prompt: {image_prompt}")
            else:
                logger.error(
                    "DSPy prediction missing expected fields "
                    f"(meme_text, image_prompt). Prediction: {prediction}"
                )
                # Fall through to fallback logic
        except Exception as e:
            logger.error(
                f"DSPy forward pass failed: {e}. Using fallback.",
                exc_info=True,  # Include traceback
            )
            # Fall through to fallback logic

        # Fallback logic (if text/image_prompt are still None)
        if text is None or image_prompt is None:
            logger.warning(f"Using fallback generation for topic: {topic}")
            base_text = random.choice(self.fallback_funny_texts)
            text = f"{base_text} - {topic}"
            image_prompt = f"A funny meme about {topic} showing {base_text.lower()}, digital art, {format} style"
            logger.info(f"Fallback generated text: {text}")
            logger.info(f"Fallback generated image_prompt: {image_prompt}")

        return text, image_prompt


class TrendPredictor(dspy.Module):
    """DSPy module for predicting meme trends."""
    
    def __init__(self) -> None:
        """Initialize the TrendPredictor."""
        super().__init__()
        # Ensure DSPy is configured before initializing ChainOfThought
        if ensure_dspy_configured():
            self.predict_trends = dspy.ChainOfThought(dspy.Signature({
                "current_trends": dspy.InputField(desc="List of current trending topics"),
                "predicted_trends": dspy.OutputField(desc="List of predicted upcoming meme trends"),
                "rationale": dspy.OutputField(desc="Explanation for the trend predictions")
            }))
        else:
            logger.error("Cannot initialize TrendPredictor: DSPy not configured.")
            self.predict_trends = None # Indicate that it's not ready
    
    def forward(self, current_trends: List[str]) -> Optional[Dict[str, Any]]:
        """
        Predict upcoming meme trends based on current trends.
        
        Args:
            current_trends: List of current trending topics
            
        Returns:
            Dictionary containing predicted trends and rationale, or None if not configured.
        """
        if not self.predict_trends:
             logger.warning("TrendPredictor not configured, cannot make predictions.")
             return None
             
        try:
            result = self.predict_trends(current_trends=", ".join(current_trends))
            
            return {
                "predicted_trends": [
                    trend.strip()
                    for trend in result.predicted_trends.split(",")
                ],
                "rationale": result.rationale
            }
        except Exception as e:
            logger.error(f"Trend prediction failed: {e}")
            return None 