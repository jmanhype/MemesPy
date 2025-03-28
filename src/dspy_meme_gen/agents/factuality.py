"""Agent for verifying factual claims in meme content."""

from typing import Dict, Any, List, TypedDict
import dspy

class FactCheck(TypedDict):
    """Result of checking a single factual claim."""
    claim: str
    is_factual: bool
    confidence: float
    correction: str

class FactualityResult(TypedDict):
    """Overall result of factuality verification."""
    is_factual: bool
    confidence: float
    factual_issues: List[str]
    corrections: List[str]

class FactualityAgent(dspy.Module):
    """
    Agent for verifying factual claims in meme content.
    Extracts claims and verifies their accuracy.
    """

    def __init__(self):
        super().__init__()
        self.claim_extractor = dspy.ChainOfThought(
            "concept: str -> claims: List[str]"
        )
        self.fact_checker = dspy.ChainOfThought(
            "claim: str -> is_factual: bool, confidence: float, correction: str"
        )
        self.assessment_generator = dspy.ChainOfThought(
            "fact_checks: List[FactCheck] -> is_factual: bool, confidence: float, factual_issues: List[str], corrections: List[str]"
        )

    def forward(self, concept: str) -> FactualityResult:
        """
        Verify factual claims in a meme concept.

        Args:
            concept: The meme concept to verify

        Returns:
            FactualityResult containing verification results
        """
        try:
            # Extract claims from concept
            claims_result = self.claim_extractor(concept=concept)
            claims = claims_result.claims

            if not claims:
                return FactualityResult(
                    is_factual=True,
                    confidence=1.0,
                    factual_issues=[],
                    corrections=[]
                )

            # Check each claim
            fact_checks: List[FactCheck] = []
            for claim in claims:
                check_result = self.fact_checker(claim=claim)
                fact_checks.append(FactCheck(
                    claim=claim,
                    is_factual=check_result.is_factual,
                    confidence=check_result.confidence,
                    correction=check_result.correction
                ))

            # Generate overall assessment
            assessment = self.assessment_generator(fact_checks=fact_checks)
            return FactualityResult(
                is_factual=assessment.is_factual,
                confidence=assessment.confidence,
                factual_issues=assessment.factual_issues,
                corrections=assessment.corrections
            )

        except Exception as e:
            raise RuntimeError(f"Failed to verify factual claims: {str(e)}") 