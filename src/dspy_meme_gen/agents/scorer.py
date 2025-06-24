"""Scoring agent for evaluating generated memes."""

from typing import Dict, Any, List
from dataclasses import dataclass
import dspy


@dataclass
class MemeScore:
    """Represents the score and feedback for a meme."""

    humor_score: float
    clarity_score: float
    creativity_score: float
    shareability_score: float
    final_score: float
    feedback: str


class ScoringAgent(dspy.Module):
    """
    Agent for evaluating and scoring generated memes.
    Considers humor, clarity, creativity, and shareability.
    """

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(
            "caption: str, image_url: str -> humor_score: float, clarity_score: float, creativity_score: float, shareability_score: float, feedback: str"
        )

    def forward(
        self,
        meme_candidates: List[Dict[str, Any]],
        factuality_check: Dict[str, Any] = None,
        instruction_check: Dict[str, Any] = None,
        appropriateness_check: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score a list of meme candidates.

        Args:
            meme_candidates: List of meme candidates to score
            factuality_check: Optional factuality verification results
            instruction_check: Optional instruction adherence results
            appropriateness_check: Optional appropriateness check results

        Returns:
            List of meme candidates with scores, sorted by final score
        """
        scored_memes = []

        for meme in meme_candidates:
            # Get base scores from LLM
            score_result = self.scorer(caption=meme["caption"], image_url=meme["image_url"])

            # Calculate weighted component score
            weighted_score = (
                (score_result.humor_score * 0.35)
                + (score_result.clarity_score * 0.25)
                + (score_result.creativity_score * 0.20)
                + (score_result.shareability_score * 0.20)
            )

            # Apply verification factors if available
            if factuality_check:
                factuality_factor = self._get_factuality_factor(factuality_check)
                weighted_score *= factuality_factor

            if instruction_check:
                instruction_factor = self._get_instruction_factor(instruction_check)
                weighted_score *= instruction_factor

            if appropriateness_check:
                appropriateness_factor = self._get_appropriateness_factor(appropriateness_check)
                weighted_score *= appropriateness_factor

            # Create MemeScore object
            meme_score = MemeScore(
                humor_score=score_result.humor_score,
                clarity_score=score_result.clarity_score,
                creativity_score=score_result.creativity_score,
                shareability_score=score_result.shareability_score,
                final_score=weighted_score,
                feedback=score_result.feedback,
            )

            # Add scores to meme info
            scored_meme = meme.copy()
            scored_meme["score"] = meme_score
            scored_memes.append(scored_meme)

        # Sort by score descending
        scored_memes.sort(key=lambda x: x["score"].final_score, reverse=True)
        return scored_memes

    def _get_factuality_factor(self, factuality_check: Dict[str, Any]) -> float:
        """Calculate adjustment factor based on factuality check."""
        if factuality_check["is_factual"]:
            return 1.0
        elif (
            factuality_check.get("factual_issues", [])
            and len(factuality_check["factual_issues"]) < 2
        ):
            return 0.7  # Minor issues
        else:
            return 0.3  # Major issues

    def _get_instruction_factor(self, instruction_check: Dict[str, Any]) -> float:
        """Calculate adjustment factor based on instruction adherence."""
        if instruction_check["constraints_met"]:
            return 1.0
        elif instruction_check.get("violations", []) and len(instruction_check["violations"]) < 2:
            return 0.8  # Minor deviations
        else:
            return 0.5  # Major deviations

    def _get_appropriateness_factor(self, appropriateness_check: Dict[str, Any]) -> float:
        """Calculate adjustment factor based on appropriateness check."""
        return 1.0 if appropriateness_check["is_appropriate"] else 0.0
