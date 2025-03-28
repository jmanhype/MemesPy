"""Main DSPy pipeline for meme generation."""

from typing import Dict, List, Optional, TypedDict, Any
import logging
from dataclasses import dataclass
from fastapi import HTTPException

import dspy
from .agents.router import RouterAgent
from .agents.trend_scanner import TrendScanningAgent
from .agents.format_selector import FormatSelectionAgent
from .agents.prompt_generator import PromptGenerationAgent
from .agents.image_renderer import ImageRenderingAgent
from .agents.factuality import FactualityAgent
from .agents.instruction_following import InstructionFollowingAgent
from .agents.appropriateness import AppropriatenessAgent
from .agents.scoring import ScoringAgent
from .agents.refinement import RefinementLoopAgent, RefinementConfig
from .exceptions.meme_specific import PipelineError
from .models.content_guidelines import ContentGuideline

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the meme generation pipeline."""
    quality_threshold: float = 0.7
    max_variants: int = 3
    enable_refinement: bool = True
    enable_verification: bool = True
    enable_trend_analysis: bool = True

class GenerationResult(TypedDict):
    """Result of the meme generation pipeline."""
    status: str  # 'success' or 'error'
    meme: Optional[Dict[str, Any]]  # Best meme if successful
    alternatives: List[Dict[str, Any]]  # Alternative memes
    error: Optional[str]  # Error message if failed
    metrics: Dict[str, Any]  # Performance metrics

class MemeGenerationPipeline(dspy.Module):
    """Main pipeline for meme generation.
    
    Attributes:
        router: Router agent for determining generation pathway
        trend_scanner: Agent for scanning current trends
        format_selector: Agent for selecting meme format
        prompt_generator: Agent for generating prompts
        image_renderer: Agent for rendering images
        factuality_agent: Agent for checking factual accuracy
        instruction_agent: Agent for checking instruction adherence
        appropriateness_agent: Agent for checking content appropriateness
        scorer: Agent for scoring memes
        refinement_agent: Agent for refining memes
        quality_threshold: Minimum quality threshold for memes
        content_guidelines: Content guidelines for meme generation
    """
    
    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        quality_threshold: float = 0.7,
        content_guidelines: Optional[ContentGuideline] = None,
    ) -> None:
        """Initialize the pipeline.
        
        Args:
            api_keys: Optional API keys for external services
            quality_threshold: Minimum quality threshold for memes
            content_guidelines: Optional content guidelines
        """
        super().__init__()
        self.router = RouterAgent()
        self.trend_scanner = TrendScanningAgent()
        self.format_selector = FormatSelectionAgent()
        self.prompt_generator = PromptGenerationAgent()
        self.image_renderer = ImageRenderingAgent(api_key=api_keys.get("openai") if api_keys else None)
        self.factuality_agent = FactualityAgent()
        self.instruction_agent = InstructionFollowingAgent()
        self.appropriateness_agent = AppropriatenessAgent()
        self.scorer = ScoringAgent()
        self.refinement_agent = RefinementLoopAgent()
        self.quality_threshold = quality_threshold
        self.content_guidelines = content_guidelines or ContentGuideline()
        
    def forward(self, user_request: str) -> Dict[str, Any]:
        """Generate a meme based on user request.
        
        Args:
            user_request: User's meme generation request
            
        Returns:
            Dictionary containing generated meme and metadata
            
        Raises:
            HTTPException: If meme generation fails
        """
        try:
            # Router determines the generation pathway
            generation_plan = self.router(user_request)
            
            # Trend scanning (if needed)
            if generation_plan.get("needs_trend_data", False):
                trends = self.trend_scanner(generation_plan.get("topic"))
            else:
                trends = None
                
            # Format selection based on request and trends
            format_selection = self.format_selector(
                generation_plan.get("topic"),
                trends
            )
            
            # Generate initial meme concept
            meme_concept = {
                "topic": generation_plan.get("topic"),
                "format": format_selection.get("selected_format")
            }
            
            # Verification pre-checks
            if generation_plan.get("verification_needs", {}).get("factuality", False):
                factuality_check = self.factuality_agent(
                    meme_concept,
                    generation_plan.get("topic")
                )
            else:
                factuality_check = {"is_factual": True}
                
            if generation_plan.get("verification_needs", {}).get("instructions", False):
                instruction_check = self.instruction_agent(
                    meme_concept,
                    generation_plan.get("constraints", {})
                )
            else:
                instruction_check = {"constraints_met": True}
                
            appropriateness_check = self.appropriateness_agent(meme_concept)
            
            # Only proceed if appropriate
            if not appropriateness_check.get("is_appropriate", True):
                return {
                    "status": "rejected",
                    "reason": "Content deemed inappropriate",
                    "suggestion": appropriateness_check.get("alternatives", [])
                }
            
            # Generate prompts
            prompts = self.prompt_generator(
                generation_plan.get("topic"),
                format_selection,
                generation_plan.get("constraints")
            )
            
            # Generate image variants (can be multiple)
            meme_candidates = []
            for _ in range(generation_plan.get("num_variants", 1)):
                meme_result = self.image_renderer(
                    prompts.get("image_prompt"),
                    prompts.get("caption"),
                    format_selection.get("format_details")
                )
                
                meme_candidate = {
                    "caption": prompts.get("caption"),
                    "image_prompt": prompts.get("image_prompt"),
                    "image_url": meme_result.get("image_url"),
                    "topic": generation_plan.get("topic"),
                    "format": format_selection.get("selected_format")
                }
                
                meme_candidates.append(meme_candidate)
            
            # Score and rank candidates
            scored_memes = self.scorer(
                meme_candidates,
                factuality_check,
                instruction_check,
                appropriateness_check
            )
            
            # Refinement if needed
            best_meme = scored_memes[0]
            if best_meme.get("final_score", 0) < self.quality_threshold:
                refined_meme = self.refinement_agent(
                    scored_memes,
                    user_request,
                    format_selection
                )
                return {
                    "status": "success",
                    "meme": refined_meme,
                    "alternatives": scored_memes[:3]  # Return top 3 alternatives
                }
            
            return {
                "status": "success",
                "meme": best_meme,
                "alternatives": scored_memes[1:3]  # Return top 3 alternatives excluding the best
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate meme: {str(e)}"
            ) 