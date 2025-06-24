"""Format selection agent for choosing appropriate meme templates."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import dspy
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from dspy_meme_gen.models.meme import MemeTemplate


@dataclass
class FormatSelectionResult:
    """Result of format selection process."""

    selected_format: str
    format_details: Dict[str, Any]
    rationale: str
    confidence_score: float


class FormatSelectionAgent(dspy.Module):
    """Agent for selecting the most appropriate meme format based on topic and trends."""

    def __init__(self, db_session: AsyncSession) -> None:
        """
        Initialize the format selection agent.

        Args:
            db_session: SQLAlchemy async session for database access
        """
        super().__init__()
        self.db_session = db_session
        self.format_matcher = dspy.ChainOfThought(
            "Given a meme topic {topic} and trending information {trends}, select the most appropriate meme format."
        )

    async def get_available_formats(self) -> List[Dict[str, Any]]:
        """
        Retrieve available meme formats from database.

        Returns:
            List of meme format dictionaries
        """
        query = select(MemeTemplate).order_by(MemeTemplate.popularity_score.desc())
        result = await self.db_session.execute(query)
        templates = result.scalars().all()
        return [template.to_dict() for template in templates]

    async def forward(
        self,
        topic: str,
        trends: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> FormatSelectionResult:
        """
        Select the most appropriate meme format.

        Args:
            topic: The meme topic
            trends: Optional trending information
            constraints: Optional format constraints

        Returns:
            FormatSelectionResult containing selected format and rationale
        """
        # Get available formats
        formats = await self.get_available_formats()

        # Match topic with appropriate format using LLM
        format_result = self.format_matcher(topic=topic, trends=trends if trends else {})

        # Find the best matching format from available ones
        selected_format = None
        highest_score = 0.0

        for fmt in formats:
            # Skip formats that don't meet constraints
            if constraints and not self._meets_constraints(fmt, constraints):
                continue

            # Calculate match score based on format matcher output
            match_score = self._calculate_match_score(
                fmt, format_result.format_characteristics, format_result.topic_requirements
            )

            if match_score > highest_score:
                highest_score = match_score
                selected_format = fmt

        if not selected_format:
            # If no format meets constraints, use the most popular one
            selected_format = formats[0]
            rationale = "No format perfectly matches constraints. Selected most popular format."
            confidence_score = 0.5
        else:
            rationale = format_result.rationale
            confidence_score = highest_score

        return FormatSelectionResult(
            selected_format=selected_format["name"],
            format_details=selected_format,
            rationale=rationale,
            confidence_score=confidence_score,
        )

    def _meets_constraints(self, format_dict: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """
        Check if a format meets given constraints.

        Args:
            format_dict: Format dictionary to check
            constraints: Constraints to verify

        Returns:
            True if format meets constraints, False otherwise
        """
        if "aspect_ratio" in constraints:
            if format_dict.get("aspect_ratio") != constraints["aspect_ratio"]:
                return False

        if "style" in constraints:
            if constraints["style"].lower() not in format_dict.get("style", "").lower():
                return False

        if "text_placement" in constraints:
            if constraints["text_placement"] not in format_dict.get("text_positions", []):
                return False

        return True

    def _calculate_match_score(
        self,
        format_dict: Dict[str, Any],
        format_characteristics: Dict[str, float],
        topic_requirements: Dict[str, float],
    ) -> float:
        """
        Calculate how well a format matches the requirements.

        Args:
            format_dict: Format to evaluate
            format_characteristics: Required format characteristics
            topic_requirements: Topic-specific requirements

        Returns:
            Match score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0

        # Check format characteristics
        for char, weight in format_characteristics.items():
            if char in format_dict and format_dict[char]:
                score += weight
            total_weight += weight

        # Check topic requirements
        for req, weight in topic_requirements.items():
            if req in format_dict.get("tags", []):
                score += weight
            total_weight += weight

        # Add popularity bonus (max 0.2)
        popularity_bonus = min(format_dict.get("popularity_score", 0.0) * 0.2, 0.2)

        # Normalize score
        if total_weight > 0:
            return (score / total_weight) * 0.8 + popularity_bonus
        return popularity_bonus
