"""Agent for verifying factual claims in meme content."""

from typing import Dict, Any, List, TypedDict, Optional
import dspy


class FactCheck(TypedDict):
    """Result of checking a single factual claim."""

    claim: str
    is_factual: bool
    confidence: float
    evidence: List[str]
    correction: Optional[str]


class FactualityAgent(dspy.Module):
    """
    Agent for verifying factual claims in meme content.
    Extracts claims and verifies their accuracy.
    """

    def __init__(self):
        super().__init__()
        self.claim_extractor = dspy.ChainOfThought("concept: str -> claims: List[str]")
        self.fact_checker = dspy.ChainOfThought(
            "claim: str -> is_factual: bool, confidence: float, evidence: List[str]"
        )
        self.correction_generator = dspy.ChainOfThought("claim: str -> correction: str")

    def forward(
        self, concept: Dict[str, Any], topic: str, strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Verify factual claims in a meme concept.

        Args:
            concept: The meme concept to verify
            topic: The topic of the meme
            strict_mode: Whether to use strict verification

        Returns:
            Dict containing verification results
        """
        try:
            # Extract claims from concept
            concept_str = f"{concept.get('topic', '')} - {concept.get('caption', '')} ({concept.get('format', '')})"
            claims_result = self.claim_extractor(concept=concept_str)
            claims = getattr(claims_result, "claims", [])

            if not claims:
                return {
                    "is_factual": True,
                    "confidence": 1.0,
                    "checks": [],
                    "suggested_corrections": {},
                    "overall_assessment": "No factual claims identified - content appears to be opinion/humor based",
                }

            # Check each claim
            fact_checks: List[FactCheck] = []
            corrections = {}

            for claim in claims:
                check_result = self.fact_checker(claim=claim)
                is_factual = getattr(check_result, "is_factual", True)
                confidence = getattr(check_result, "confidence", 0.5)
                evidence = getattr(check_result, "evidence", [])

                fact_check = FactCheck(
                    claim=claim,
                    is_factual=is_factual,
                    confidence=confidence,
                    evidence=evidence,
                    correction=None,
                )

                # Generate correction if claim is not factual
                if not is_factual:
                    correction_result = self.correction_generator(claim=claim)
                    correction = getattr(correction_result, "correction", None)
                    if correction:
                        fact_check["correction"] = correction
                        corrections[claim] = correction

                fact_checks.append(fact_check)

            # Generate overall assessment
            assessment = self._generate_assessment(fact_checks, strict_mode)

            # Calculate overall factuality
            if fact_checks:
                factual_count = sum(1 for check in fact_checks if check["is_factual"])
                accuracy = factual_count / len(fact_checks)
                is_factual = accuracy >= (0.9 if strict_mode else 0.75)
                confidence = sum(check["confidence"] for check in fact_checks) / len(fact_checks)
            else:
                is_factual = True
                confidence = 1.0

            return {
                "is_factual": is_factual,
                "confidence": confidence,
                "checks": fact_checks,
                "suggested_corrections": corrections,
                "overall_assessment": assessment,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to verify factuality: {str(e)}")

    def _generate_assessment(self, checks: List[FactCheck], strict_mode: bool = False) -> str:
        """Generate an overall assessment of factuality."""
        if not checks:
            return "No factual claims identified - content appears to be opinion/humor based"

        factual_count = sum(1 for check in checks if check["is_factual"])
        total_count = len(checks)
        accuracy = (factual_count / total_count) * 100

        threshold = 90 if strict_mode else 75

        if accuracy == 100:
            return f"All claims verified as factual with high confidence (threshold: {threshold}%)"
        elif accuracy >= threshold:
            return (
                f"Content is {accuracy:.1f}% accurate and meets factuality threshold ({threshold}%)"
            )
        else:
            return f"Content has significant inaccuracies ({accuracy:.1f}% accurate, threshold: {threshold}%)"
