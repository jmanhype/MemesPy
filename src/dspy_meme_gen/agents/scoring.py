"""Scoring Agent for evaluating meme quality."""

from typing import Dict, Any, List, Optional, TypedDict, Union
import logging
from dataclasses import dataclass

import dspy
from ..exceptions.meme_specific import ScoringError
from ..models.content_guidelines import ContentGuideline

logger = logging.getLogger(__name__)


@dataclass
class ComponentScores:
    """Component-wise scores for a meme."""

    humor_score: float
    clarity_score: float
    creativity_score: float
    shareability_score: float
    factuality_score: Optional[float] = None
    appropriateness_score: Optional[float] = None


class ScoringResult(TypedDict):
    """Result of meme scoring including component scores and final score."""

    final_score: float
    component_scores: ComponentScores
    feedback: List[str]
    improvement_suggestions: List[str]


class ScoringAgent(dspy.Module):
    """Agent for evaluating meme quality using multiple criteria."""

    def __init__(self, content_guidelines: Optional[ContentGuideline] = None):
        """Initialize the ScoringAgent.

        Args:
            content_guidelines: Optional guidelines for content evaluation
        """
        super().__init__()
        self.content_guidelines = content_guidelines or ContentGuideline()

        # Initialize DSPy predictors for different scoring aspects
        self.humor_scorer = dspy.ChainOfThought("caption, image_description -> score, reasoning")

        self.clarity_scorer = dspy.ChainOfThought("caption, image_description -> score, reasoning")

        self.creativity_scorer = dspy.ChainOfThought(
            "caption, image_description -> score, reasoning"
        )

        self.shareability_scorer = dspy.ChainOfThought(
            "caption, image_description -> score, reasoning"
        )

    def forward(
        self,
        caption: str,
        image_description: str,
        factuality_check: Optional[Dict] = None,
        appropriateness_check: Optional[Dict] = None,
    ) -> ScoringResult:
        """Score a meme based on multiple criteria.

        Args:
            caption: The meme's caption text
            image_description: Description of the meme image
            factuality_check: Optional factuality check results
            appropriateness_check: Optional appropriateness check results

        Returns:
            ScoringResult containing final score, component scores, and feedback

        Raises:
            ScoringError: If scoring fails for any component
        """
        try:
            # Get component scores
            humor_result = self.humor_scorer(caption=caption, image_description=image_description)
            clarity_result = self.clarity_scorer(
                caption=caption, image_description=image_description
            )
            creativity_result = self.creativity_scorer(
                caption=caption, image_description=image_description
            )
            shareability_result = self.shareability_scorer(
                caption=caption, image_description=image_description
            )

            # Extract scores and reasoning
            component_scores = ComponentScores(
                humor_score=float(humor_result.score),
                clarity_score=float(clarity_result.score),
                creativity_score=float(creativity_result.score),
                shareability_score=float(shareability_result.score),
            )

            feedback = [
                f"Humor: {humor_result.reasoning}",
                f"Clarity: {clarity_result.reasoning}",
                f"Creativity: {creativity_result.reasoning}",
                f"Shareability: {shareability_result.reasoning}",
            ]

            # Apply verification factors if available
            final_score = self._calculate_final_score(
                component_scores, factuality_check, appropriateness_check
            )

            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                component_scores,
                humor_result.reasoning,
                clarity_result.reasoning,
                creativity_result.reasoning,
                shareability_result.reasoning,
            )

            return {
                "final_score": final_score,
                "component_scores": component_scores,
                "feedback": feedback,
                "improvement_suggestions": improvement_suggestions,
            }

        except Exception as e:
            logger.error(f"Error scoring meme: {str(e)}")
            raise ScoringError(
                message="Failed to score meme", details={"caption": caption, "error": str(e)}
            )

    def _calculate_final_score(
        self,
        scores: ComponentScores,
        factuality_check: Optional[Dict] = None,
        appropriateness_check: Optional[Dict] = None,
    ) -> float:
        """Calculate final score with weights and verification factors.

        Args:
            scores: Component scores
            factuality_check: Optional factuality check results
            appropriateness_check: Optional appropriateness check results

        Returns:
            Final weighted score between 0 and 1
        """
        # Base weights for components
        weights = {"humor": 0.35, "clarity": 0.25, "creativity": 0.20, "shareability": 0.20}

        # Calculate base score
        base_score = (
            scores.humor_score * weights["humor"]
            + scores.clarity_score * weights["clarity"]
            + scores.creativity_score * weights["creativity"]
            + scores.shareability_score * weights["shareability"]
        )

        # Apply verification factors
        if factuality_check:
            factuality_factor = 1.0 if factuality_check.get("is_factual", True) else 0.7
            base_score *= factuality_factor

        if appropriateness_check:
            appropriateness_factor = (
                1.0 if appropriateness_check.get("is_appropriate", True) else 0.5
            )
            base_score *= appropriateness_factor

        return round(base_score, 2)

    def _generate_improvement_suggestions(
        self,
        scores: ComponentScores,
        humor_reasoning: str,
        clarity_reasoning: str,
        creativity_reasoning: str,
        shareability_reasoning: str,
    ) -> List[str]:
        """Generate improvement suggestions based on scores and reasoning.

        Args:
            scores: Component scores
            *_reasoning: Reasoning for each component score

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Add suggestions for low-scoring components
        if scores.humor_score < 0.7:
            suggestions.append(f"Improve humor: {self._extract_suggestion(humor_reasoning)}")
        if scores.clarity_score < 0.7:
            suggestions.append(f"Enhance clarity: {self._extract_suggestion(clarity_reasoning)}")
        if scores.creativity_score < 0.7:
            suggestions.append(
                f"Boost creativity: {self._extract_suggestion(creativity_reasoning)}"
            )
        if scores.shareability_score < 0.7:
            suggestions.append(
                f"Increase shareability: {self._extract_suggestion(shareability_reasoning)}"
            )

        return suggestions

    def _extract_suggestion(self, reasoning: str) -> str:
        """Extract actionable suggestion from reasoning text."""
        # TODO: Implement more sophisticated suggestion extraction
        return reasoning.split(".")[-2] if len(reasoning.split(".")) > 1 else reasoning
