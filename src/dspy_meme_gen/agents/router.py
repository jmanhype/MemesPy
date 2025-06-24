"""Router Agent for analyzing and routing meme generation requests."""

from typing import Dict, Any
import dspy
from loguru import logger


class RouteResult(Dict[str, Any]):
    """Type definition for router results."""

    topic: str
    format: str
    verification_needs: Dict[str, bool]
    constraints: Dict[str, Any]
    generation_approach: str


class RouterAgent(dspy.Module):
    """
    Agent for routing and analyzing meme generation requests.
    Determines the appropriate verification agents and generation approach.
    """

    def __init__(self) -> None:
        """Initialize the router agent."""
        super().__init__()
        self.router = dspy.ChainOfThought(
            "request: str -> topic: str, format: str, verification_needs: dict, constraints: dict, generation_approach: str"
        )

    def forward(self, user_request: str) -> Dict[str, Any]:
        """
        Route a user request to determine generation approach and requirements.

        Args:
            user_request: The user's meme generation request

        Returns:
            Dict containing routing information including topic, format, verification needs,
            constraints, and generation approach
        """
        try:
            logger.info(f"Processing routing request: {user_request}")

            route_result = self.router(request=user_request)

            result: RouteResult = {
                "topic": route_result.topic,
                "format": route_result.format,
                "verification_needs": route_result.verification_needs,
                "constraints": route_result.constraints,
                "generation_approach": route_result.generation_approach,
            }

            logger.debug(f"Route result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in router agent: {str(e)}")
            raise RuntimeError(f"Failed to process routing request: {str(e)}")
