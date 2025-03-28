"""
Meme generator module using DSPy.
"""

import logging
import random
from typing import List, Optional, Dict, Any, Tuple

import dspy

from .config import ensure_dspy_configured

logger = logging.getLogger(__name__)


class MemeSignature(dspy.Signature):
    """
    Signature for meme generation.
    
    This signature defines the inputs and outputs for the meme generation task.
    """
    
    topic: str = dspy.InputField(desc="The topic or theme for the meme")
    format: str = dspy.InputField(desc="The meme format to use (e.g., 'standard', 'modern', 'comparison')")
    context: Optional[str] = dspy.InputField(desc="Additional context or requirements for the meme")
    
    text: str = dspy.OutputField(desc="The text content for the meme")
    image_prompt: str = dspy.OutputField(desc="A detailed prompt for image generation")
    rationale: str = dspy.OutputField(desc="Explanation of why this meme would be effective")
    score: float = dspy.OutputField(desc="A score between 0 and 1 indicating the predicted effectiveness")


class MemeGenerator(dspy.Module):
    """
    DSPy module for generating memes.
    
    This module generates meme text and image prompts based on the given topic,
    format, and context using the DSPy framework.
    """
    
    def __init__(self) -> None:
        """
        Initialize the MemeGenerator.
        
        Sets up the DSPy module with the appropriate signature for meme generation.
        """
        super().__init__()
        ensure_dspy_configured()
        self.generate_meme = dspy.ChainOfThought(MemeSignature)
    
    def forward(self, topic: str, format: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a meme based on the given topic and format.
        
        Args:
            topic: The topic or theme for the meme.
            format: The meme format to use.
            context: Optional additional context or requirements.
            
        Returns:
            Dictionary containing the meme text, image prompt, rationale, and score.
        """
        try:
            result = self.generate_meme(
                topic=topic,
                format=format,
                context=context
            )
            
            # Convert to dictionary for easier handling in API
            return {
                "text": result.text,
                "image_prompt": result.image_prompt,
                "rationale": result.rationale,
                "score": result.score
            }
        except Exception as e:
            logger.error(f"Error generating meme with DSPy: {e}")
            return self.fallback_generation(topic, format, context)
    
    def fallback_generation(self, topic: str, format: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback method for meme generation when DSPy fails.
        
        Args:
            topic: The topic or theme for the meme.
            format: The meme format to use.
            context: Optional additional context or requirements.
            
        Returns:
            Dictionary containing fallback meme text, image prompt, rationale, and score.
        """
        logger.warning("Using fallback meme generation mechanism")
        
        # Simple templates for fallback generation
        templates = [
            "When you try to {action} but {consequence}",
            "Nobody:\nAbsolutely nobody:\nMe: {action}",
            "{subject}: exists\n{reaction}: I'm about to end this man's whole career",
            "What I think I look like: {ideal}\nWhat I actually look like: {reality}",
            "{first_part}\n{second_part}\nProfit!",
        ]
        
        # Generate some basic text based on the topic
        action = f"use {topic}"
        consequence = "it fails spectacularly"
        subject = topic
        reaction = "Everyone else"
        ideal = f"Expert {topic} user"
        reality = f"Confused {topic} noob"
        first_part = f"Step 1: Learn {topic}"
        second_part = f"Step 2: ?"
        
        # Choose a template based on the format
        if format == "comparison":
            text = templates[3].format(ideal=ideal, reality=reality)
        elif format == "steps":
            text = templates[4].format(first_part=first_part, second_part=second_part)
        else:
            text = random.choice(templates).format(
                action=action, 
                consequence=consequence,
                subject=subject,
                reaction=reaction,
                ideal=ideal,
                reality=reality,
                first_part=first_part,
                second_part=second_part
            )
        
        # Generate a basic image prompt
        image_prompt = f"A meme about {topic}, showing {action} with {consequence}"
        
        return {
            "text": text,
            "image_prompt": image_prompt,
            "rationale": f"This meme uses a classic template to highlight common experiences with {topic}.",
            "score": 0.6  # Moderate score for fallback generation
        } 