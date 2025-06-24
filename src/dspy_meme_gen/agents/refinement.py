"""Refinement Loop Agent for iterative meme improvement."""

from typing import Dict, List, Optional, TypedDict, Any
import logging
from dataclasses import dataclass

import dspy
from .scoring import ScoringAgent, ScoringResult
from .prompt_generator import PromptGenerationAgent
from .image_renderer import ImageRenderingAgent
from ..exceptions.meme_specific import RefinementError

logger = logging.getLogger(__name__)


@dataclass
class RefinementConfig:
    """Configuration for the refinement process."""

    max_iterations: int = 3
    score_threshold: float = 0.85
    min_improvement: float = 0.05


class RefinementResult(TypedDict):
    """Result of the refinement process."""

    original_meme: Dict[str, Any]
    refined_meme: Dict[str, Any]
    improvement_score: float
    iterations: int
    refinement_history: List[Dict[str, Any]]


class RefinementLoopAgent(dspy.Module):
    """Agent for iteratively improving meme quality based on feedback."""

    def __init__(
        self,
        scoring_agent: Optional[ScoringAgent] = None,
        prompt_generator: Optional[PromptGenerationAgent] = None,
        image_renderer: Optional[ImageRenderingAgent] = None,
        config: Optional[RefinementConfig] = None,
    ):
        """Initialize the RefinementLoopAgent.

        Args:
            scoring_agent: Agent for evaluating meme quality
            prompt_generator: Agent for generating prompts
            image_renderer: Agent for rendering images
            config: Refinement process configuration
        """
        super().__init__()
        self.scoring_agent = scoring_agent or ScoringAgent()
        self.prompt_generator = prompt_generator or PromptGenerationAgent()
        self.image_renderer = image_renderer or ImageRenderingAgent()
        self.config = config or RefinementConfig()

        # Initialize DSPy predictor for planning improvements
        self.improvement_planner = dspy.ChainOfThought(
            "Given a meme with scores {scores} and feedback {feedback}, "
            "plan specific improvements for the caption and image prompt "
            "to address the identified issues."
        )

    def forward(
        self,
        meme: Dict[str, Any],
        format_details: Dict[str, Any],
        factuality_check: Optional[Dict] = None,
        appropriateness_check: Optional[Dict] = None,
    ) -> RefinementResult:
        """Iteratively refine a meme to improve its quality.

        Args:
            meme: Original meme data including caption and image
            format_details: Details about the meme format
            factuality_check: Optional factuality check results
            appropriateness_check: Optional appropriateness check results

        Returns:
            RefinementResult containing original and refined memes with history

        Raises:
            RefinementError: If refinement process fails
        """
        try:
            # Initialize tracking
            best_meme = meme.copy()
            best_score = 0.0
            refinement_history = []

            # Get initial score
            initial_score = self._score_meme(meme, factuality_check, appropriateness_check)
            best_score = initial_score["final_score"]

            for iteration in range(self.config.max_iterations):
                logger.info(f"Starting refinement iteration {iteration + 1}")

                # Plan improvements based on current scores and feedback
                improvements = self.improvement_planner(
                    scores=initial_score["component_scores"], feedback=initial_score["feedback"]
                )

                # Generate new prompts based on improvements
                new_prompts = self.prompt_generator(
                    topic=meme.get("topic"),
                    format_details=format_details,
                    improvement_suggestions=improvements.suggestions,
                )

                # Generate new image
                new_image_result = self.image_renderer(
                    new_prompts["image_prompt"], new_prompts["caption"], format_details
                )

                # Create refined meme
                refined_meme = {
                    **meme,
                    "caption": new_prompts["caption"],
                    "image_prompt": new_prompts["image_prompt"],
                    "image_url": new_image_result["image_url"],
                }

                # Score refined meme
                refined_score = self._score_meme(
                    refined_meme, factuality_check, appropriateness_check
                )

                # Track refinement
                refinement_history.append(
                    {
                        "iteration": iteration + 1,
                        "meme": refined_meme,
                        "score": refined_score,
                        "improvements": improvements.suggestions,
                    }
                )

                # Update best if improved significantly
                score_improvement = refined_score["final_score"] - best_score
                if score_improvement > self.config.min_improvement:
                    best_meme = refined_meme
                    best_score = refined_score["final_score"]
                    logger.info(f"Found better meme with score {best_score}")

                # Check if we've reached desired quality
                if best_score >= self.config.score_threshold:
                    logger.info("Reached quality threshold, stopping refinement")
                    break

                # Check if we're not improving enough
                if score_improvement < self.config.min_improvement:
                    logger.info("Insufficient improvement, stopping refinement")
                    break

            return {
                "original_meme": meme,
                "refined_meme": best_meme,
                "improvement_score": best_score - initial_score["final_score"],
                "iterations": len(refinement_history),
                "refinement_history": refinement_history,
            }

        except Exception as e:
            logger.error(f"Error during meme refinement: {str(e)}")
            raise RefinementError(
                message="Failed to refine meme", details={"original_meme": meme, "error": str(e)}
            )

    def _score_meme(
        self,
        meme: Dict[str, Any],
        factuality_check: Optional[Dict] = None,
        appropriateness_check: Optional[Dict] = None,
    ) -> ScoringResult:
        """Score a meme using the scoring agent.

        Args:
            meme: Meme data including caption and image
            factuality_check: Optional factuality check results
            appropriateness_check: Optional appropriateness check results

        Returns:
            ScoringResult from the scoring agent
        """
        return self.scoring_agent(
            caption=meme["caption"],
            image_description=meme.get("image_prompt", ""),
            factuality_check=factuality_check,
            appropriateness_check=appropriateness_check,
        )
