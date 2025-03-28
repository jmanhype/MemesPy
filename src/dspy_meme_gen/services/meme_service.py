"""Service for meme operations."""

from typing import List, Dict, Any, Optional
import logging
import asyncio

from ..repositories.meme_repository import meme_repository
from ..agents.trend_analyzer import trend_analyzer
from ..agents.format_generator import format_generator

# Set up logging
logger = logging.getLogger(__name__)


class MemeService:
    """Service for meme operations."""
    
    def __init__(self):
        """Initialize the meme service."""
        logger.info("Initializing meme service")
        self.repository = meme_repository
    
    async def get_memes(self) -> List[Dict[str, Any]]:
        """
        Get all memes.
        
        Returns:
            List of memes
        """
        logger.info("Getting all memes")
        return await self.repository.get_memes()
    
    async def get_meme(self, meme_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a meme by ID.
        
        Args:
            meme_id: ID of the meme
            
        Returns:
            Meme data if found, None otherwise
        """
        logger.info(f"Getting meme with ID: {meme_id}")
        return await self.repository.get_meme(meme_id)
    
    async def create_meme(self, topic: str, format_id: str) -> Dict[str, Any]:
        """
        Create a new meme.
        
        Args:
            topic: Topic for the meme
            format_id: Format ID for the meme
            
        Returns:
            Created meme data
        """
        logger.info(f"Creating meme with topic: {topic}, format: {format_id}")
        
        # Generate and save the meme using DSPy
        generated_meme = await self.repository.create_meme(topic, format_id)
        
        logger.info(f"Meme created with ID: {generated_meme['id']}")
        return generated_meme
    
    async def get_trending_topics(self) -> List[Dict[str, Any]]:
        """
        Get all trending topics.
        
        Returns:
            List of trending topics
        """
        logger.info("Getting trending topics")
        return await trend_analyzer.get_trending_topics()
    
    async def get_trending_topic(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a trending topic by ID.
        
        Args:
            topic_id: ID of the trending topic
            
        Returns:
            Trending topic data if found, None otherwise
        """
        logger.info(f"Getting trending topic with ID: {topic_id}")
        return await trend_analyzer.get_trending_topic(topic_id)
    
    async def get_formats(self) -> List[Dict[str, Any]]:
        """
        Get all meme formats.
        
        Returns:
            List of meme formats
        """
        logger.info("Getting meme formats")
        return await format_generator.get_formats()
    
    async def get_format(self, format_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a meme format by ID.
        
        Args:
            format_id: ID of the meme format
            
        Returns:
            Format data if found, None otherwise
        """
        logger.info(f"Getting format with ID: {format_id}")
        return await format_generator.get_format(format_id)


# Create a singleton instance
meme_service = MemeService() 