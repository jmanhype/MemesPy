"""Instruction following agent for meme generation."""

from typing import Dict, Any
import dspy

class InstructionFollowingAgent(dspy.Module):
    """Agent for verifying adherence to user instructions.
    
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
        
        # Return constraint satisfaction assessment
        return {
            "constraints_met": check_result.constraints_met,
            "violations": check_result.violations,
            "suggestions": check_result.suggestions
        }
    
    def _apply_constraints(self, text: str, constraints: Dict[str, Any]) -> str:
        """Apply constraints to text.
        
        Args:
            text: Text to apply constraints to
            constraints: Constraints to apply
            
        Returns:
            Modified text adhering to constraints
        """
        # Apply length constraints
        if "max_length" in constraints:
            text = text[:constraints["max_length"]]
        
        # Apply style constraints
        if "style" in constraints:
            if constraints["style"] == "uppercase":
                text = text.upper()
            elif constraints["style"] == "lowercase":
                text = text.lower()
            elif constraints["style"] == "title":
                text = text.title()
        
        return text 