"""Appropriateness Agent for screening meme content."""

from typing import Dict, Any, TypedDict, Optional, List
import dspy
from loguru import logger

from ..models.content_guidelines import GuidelineRepository, SeverityLevel


class ContentFlag(TypedDict):
    """Type definition for content flags."""

    category: str
    severity: str  # "low", "medium", "high"
    description: str
    affected_elements: List[str]
    suggestion: Optional[str]


class AppropriatenessResult(TypedDict):
    """Type definition for appropriateness screening results."""

    is_appropriate: bool
    flags: List[ContentFlag]
    overall_rating: str  # "safe", "questionable", "inappropriate"
    target_audience: List[str]
    suggested_modifications: Dict[str, str]


class AppropriatenessAgent(dspy.Module):
    """
    Agent responsible for screening meme content for appropriateness.

    This agent:
    - Analyzes content for potentially sensitive material
    - Considers cultural context and implications
    - Identifies target audience suitability
    - Suggests appropriate modifications
    - Maintains content guidelines compliance
    """

    def __init__(self, guideline_repository: GuidelineRepository) -> None:
        """
        Initialize the Appropriateness Agent with necessary predictors.

        Args:
            guideline_repository: Repository for accessing content guidelines
        """
        super().__init__()

        self.guideline_repository = guideline_repository

        # Content analyzer predictor
        self.content_analyzer = dspy.ChainOfThought(
            """Given a meme concept, analyze its content for:
            1. Potentially offensive or insensitive elements
            2. Cultural appropriateness and context
            3. Age-appropriate content considerations
            4. Professional environment suitability
            
            Identify any concerning elements and suggest modifications.
            """
        )

        # Audience analyzer predictor
        self.audience_analyzer = dspy.ChainOfThought(
            """Given a meme concept and its content analysis, determine:
            1. Suitable target audiences
            2. Potential impact on different groups
            3. Cultural resonance and sensitivity
            4. Context-specific appropriateness
            
            Provide detailed audience suitability assessment.
            """
        )

    def forward(
        self,
        concept: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        strict_mode: bool = False,
    ) -> AppropriatenessResult:
        """
        Screen meme content for appropriateness.

        Args:
            concept: The meme concept to screen
            context: Optional context about intended use/audience
            strict_mode: Whether to apply stricter screening standards

        Returns:
            AppropriatenessResult containing screening details
        """
        try:
            logger.info(
                f"Screening content for appropriateness: {concept.get('topic', 'Unknown topic')}"
            )

            # Get current guidelines from database
            guidelines = self.guideline_repository.get_all_guidelines()

            # Analyze content against guidelines
            content_analysis = self.content_analyzer(
                concept=concept,
                context=context if context else {},
                guidelines=guidelines,
                strict_mode=strict_mode,
            )

            # Analyze audience suitability
            audience_analysis = self.audience_analyzer(
                concept=concept,
                content_analysis=content_analysis,
                context=context if context else {},
            )

            # Collect content flags
            flags: List[ContentFlag] = []
            for issue in content_analysis.issues:
                flag: ContentFlag = {
                    "category": issue.category,
                    "severity": issue.severity,
                    "description": issue.description,
                    "affected_elements": issue.elements,
                    "suggestion": issue.suggestion if hasattr(issue, "suggestion") else None,
                }
                flags.append(flag)

            # Determine overall appropriateness
            is_appropriate = self._evaluate_appropriateness(flags, strict_mode)

            # Generate suggested modifications
            modifications = {
                flag["affected_elements"][0]: flag["suggestion"]
                for flag in flags
                if flag["suggestion"] is not None
            }

            result: AppropriatenessResult = {
                "is_appropriate": is_appropriate,
                "flags": flags,
                "overall_rating": self._determine_rating(flags),
                "target_audience": audience_analysis.suitable_audiences,
                "suggested_modifications": modifications,
            }

            logger.debug(f"Appropriateness screening complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in appropriateness screening: {str(e)}")
            raise RuntimeError(f"Failed to screen content: {str(e)}")

    def _evaluate_appropriateness(self, flags: List[ContentFlag], strict_mode: bool) -> bool:
        """
        Evaluate overall appropriateness based on flags.

        Args:
            flags: List of content flags
            strict_mode: Whether to apply stricter standards

        Returns:
            Boolean indicating if content is appropriate
        """
        if not flags:
            return True

        # Count flags by severity
        severity_counts = {
            "high": sum(1 for flag in flags if flag["severity"] == "high"),
            "medium": sum(1 for flag in flags if flag["severity"] == "medium"),
            "low": sum(1 for flag in flags if flag["severity"] == "low"),
        }

        if strict_mode:
            # In strict mode, any high severity flag or multiple medium flags make it inappropriate
            return severity_counts["high"] == 0 and severity_counts["medium"] <= 1
        else:
            # In normal mode, multiple high severity flags or many medium flags make it inappropriate
            return severity_counts["high"] <= 1 and severity_counts["medium"] <= 2

    def _determine_rating(self, flags: List[ContentFlag]) -> str:
        """
        Determine overall content rating based on flags.

        Args:
            flags: List of content flags

        Returns:
            Rating string
        """
        if not flags:
            return "safe"

        # Check for high severity flags
        has_high = any(flag["severity"] == "high" for flag in flags)
        has_medium = any(flag["severity"] == "medium" for flag in flags)

        if has_high:
            return "inappropriate"
        elif has_medium:
            return "questionable"
        else:
            return "safe"
