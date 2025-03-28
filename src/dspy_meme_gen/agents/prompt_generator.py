"""Prompt generation agent for creating meme captions and image prompts."""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import dspy
from loguru import logger


@dataclass
class PromptGenerationResult:
    """Result of prompt generation process."""
    caption: str
    image_prompt: str
    reasoning: str
    text_positions: Dict[str, str]
    style_guide: Dict[str, Any]


class PromptGenerationAgent(dspy.Module):
    """Agent for generating optimized text captions and image prompts for memes."""

    def __init__(self) -> None:
        """Initialize the prompt generation agent."""
        super().__init__()
        self.caption_generator = dspy.ChainOfThought(
            "Given a meme topic {topic} and format {format}, generate a funny and relevant caption."
        )
        self.image_prompt_generator = dspy.ChainOfThought(
            "Given a meme topic {topic}, format {format}, and caption {caption}, create a detailed image generation prompt."
        )

    def forward(
        self,
        topic: str,
        format_details: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        style_preferences: Optional[Dict[str, Any]] = None
    ) -> PromptGenerationResult:
        """
        Generate captions and image prompts for a meme.
        
        Args:
            topic: The meme topic
            format_details: Details about the selected meme format
            constraints: Optional constraints for generation
            style_preferences: Optional style preferences
            
        Returns:
            PromptGenerationResult containing generated prompts and metadata
        """
        try:
            logger.info(f"Generating prompts for topic: {topic}")
            
            # Generate caption based on format and topic
            caption_result = self.caption_generator(
                topic=topic,
                format=format_details["name"],
                format_structure=format_details.get("structure", {}),
                constraints=constraints if constraints else {}
            )
            
            # Generate image prompt based on caption and format
            image_prompt_result = self.image_prompt_generator(
                topic=topic,
                format=format_details["name"],
                caption=caption_result.caption,
                format_structure=format_details.get("structure", {}),
                style_preferences=style_preferences if style_preferences else {}
            )
            
            # Apply any constraints to the prompts
            caption = self._apply_constraints(
                caption_result.caption,
                constraints if constraints else {},
                format_details.get("structure", {}).get("text_positions", [])
            )
            
            image_prompt = self._apply_style_preferences(
                image_prompt_result.image_prompt,
                style_preferences if style_preferences else {},
                format_details.get("structure", {}).get("style", "")
            )
            
            # Organize text positions
            text_positions = self._organize_text_positions(
                caption,
                format_details.get("structure", {}).get("text_positions", [])
            )
            
            # Compile style guide
            style_guide = self._compile_style_guide(
                format_details.get("structure", {}),
                style_preferences if style_preferences else {}
            )
            
            result = PromptGenerationResult(
                caption=caption,
                image_prompt=image_prompt,
                reasoning=caption_result.reasoning,
                text_positions=text_positions,
                style_guide=style_guide
            )
            
            logger.debug(f"Generated prompts: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prompt generation: {str(e)}")
            raise RuntimeError(f"Failed to generate prompts: {str(e)}")

    def _apply_constraints(
        self,
        text: str,
        constraints: Dict[str, Any],
        text_positions: List[str]
    ) -> str:
        """
        Apply constraints to generated text.
        
        Args:
            text: Text to modify
            constraints: Constraints to apply
            text_positions: Valid text positions
            
        Returns:
            Modified text
        """
        # Apply length constraints
        if "max_length" in constraints:
            text = text[:constraints["max_length"]]
        
        # Apply position-specific formatting
        if "text_placement" in constraints and constraints["text_placement"] in text_positions:
            # Add position-specific formatting (e.g., alignment, size)
            pass
        
        # Apply content constraints
        if "tone" in constraints:
            # Adjust tone (e.g., formal, casual, humorous)
            pass
        
        return text

    def _apply_style_preferences(
        self,
        prompt: str,
        preferences: Dict[str, Any],
        base_style: str
    ) -> str:
        """
        Apply style preferences to image prompt.
        
        Args:
            prompt: Image prompt to modify
            preferences: Style preferences to apply
            base_style: Base style from format
            
        Returns:
            Modified prompt
        """
        # Start with base style
        style_elements = [base_style] if base_style else []
        
        # Add artistic style
        if "artistic_style" in preferences:
            style_elements.append(preferences["artistic_style"])
        
        # Add color scheme
        if "color_scheme" in preferences:
            style_elements.append(f"using {preferences['color_scheme']} colors")
        
        # Add lighting
        if "lighting" in preferences:
            style_elements.append(f"with {preferences['lighting']} lighting")
        
        # Combine style elements with base prompt
        if style_elements:
            prompt = f"{prompt}, {', '.join(style_elements)}"
        
        return prompt

    def _organize_text_positions(self, text: str, positions: List[str]) -> Dict[str, str]:
        """
        Organize text into position-specific segments.
        
        Args:
            text: Text to organize
            positions: Available positions
            
        Returns:
            Dictionary mapping positions to text segments
        """
        # Split text into segments based on format requirements
        segments = text.split("|") if "|" in text else [text]
        
        # Map segments to positions
        text_map = {}
        for i, position in enumerate(positions):
            if i < len(segments):
                text_map[position] = segments[i].strip()
            else:
                text_map[position] = ""
        
        return text_map

    def _compile_style_guide(
        self,
        format_structure: Dict[str, Any],
        style_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compile a style guide for the meme.
        
        Args:
            format_structure: Format structure details
            style_preferences: Style preferences
            
        Returns:
            Style guide dictionary
        """
        style_guide = {
            "layout": format_structure.get("layout", "default"),
            "text_style": {
                "font": style_preferences.get("font", "Impact"),
                "color": style_preferences.get("text_color", "white"),
                "stroke": style_preferences.get("text_stroke", "black"),
                "alignment": format_structure.get("text_alignment", "center")
            },
            "image_style": {
                "base_style": format_structure.get("style", ""),
                "artistic_style": style_preferences.get("artistic_style", ""),
                "color_scheme": style_preferences.get("color_scheme", ""),
                "lighting": style_preferences.get("lighting", "")
            }
        }
        
        return style_guide 