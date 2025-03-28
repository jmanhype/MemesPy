"""Verification agents for meme generation."""

from typing import Dict, Any, Optional
import dspy

class FactualityAgent(dspy.Module):
    """Agent for verifying factual accuracy.
    
    Attributes:
        checker: DSPy module for checking factual accuracy
    """
    
    def __init__(self) -> None:
        """Initialize the factuality agent."""
        super().__init__()
        self.checker = dspy.ChainOfThought(
            "Given a meme concept {concept} with topic {topic}, identify and verify any factual claims."
        )
    
    def forward(self, concept: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Check factual accuracy of a meme concept.
        
        Args:
            concept: Meme concept to check
            topic: Topic of the meme
            
        Returns:
            Dictionary containing verification results
        """
        check_result = self.checker(concept=concept, topic=topic)
        
        return {
            "is_factual": check_result.is_factual,
            "confidence": check_result.confidence,
            "factual_issues": check_result.factual_issues,
            "corrections": check_result.corrections
        }

class AppropriatenessAgent(dspy.Module):
    """Agent for verifying content appropriateness.
    
    Attributes:
        screener: DSPy module for screening content
    """
    
    def __init__(self) -> None:
        """Initialize the appropriateness agent."""
        super().__init__()
        self.screener = dspy.ChainOfThought(
            "Given a meme concept {concept}, assess its appropriateness across different contexts and audiences."
        )
    
    def forward(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Check appropriateness of a meme concept.
        
        Args:
            concept: Meme concept to check
            
        Returns:
            Dictionary containing verification results
        """
        screen_result = self.screener(concept=concept)
        
        return {
            "is_appropriate": screen_result.is_appropriate,
            "concerns": screen_result.concerns,
            "alternatives": screen_result.alternatives
        }

class InstructionFollowingAgent(dspy.Module):
    """Agent for verifying instruction adherence.
    
    Attributes:
        checker: DSPy module for checking instruction adherence
    """
    
    def __init__(self) -> None:
        """Initialize the instruction following agent."""
        super().__init__()
        self.checker = dspy.ChainOfThought(
            "Given a meme concept {concept} and constraints {constraints}, verify if all constraints are satisfied."
        )
    
    def forward(self, concept: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a meme concept adheres to given constraints.
        
        Args:
            concept: Meme concept to check
            constraints: Constraints to verify against
            
        Returns:
            Dictionary containing verification results
        """
        check_result = self.checker(concept=concept, constraints=constraints)
        
        return {
            "constraints_met": check_result.constraints_met,
            "violations": check_result.violations,
            "suggestions": check_result.suggestions
        } 