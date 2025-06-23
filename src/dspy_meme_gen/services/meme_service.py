"""Service for meme operations with async event sourcing."""

from typing import List, Dict, Any, Optional
import logging
import asyncio
from uuid import UUID, uuid4

# Command definitions are now imported from CQRS module

# Set up logging
logger = logging.getLogger(__name__)

# Try to import CQRS components, fallback if not available
try:
    from ..cqrs.command_handlers import handle_command
    from ..cqrs.projections import (
        list_meme_projections, get_meme_projection, get_pipeline_status, get_daily_metrics
    )
    from ..cqrs.event_bus import publish_event
    from ..cqrs.events import MemeViewed, MemeShared
    from ..cqrs.commands.meme_commands import (
        GenerateMemeCommand, ScoreMemeCommand, DeleteMemeCommand
    )
    CQRS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CQRS components not available: {e}")
    CQRS_AVAILABLE = False
    
try:
    from ..agents.trend_analyzer import trend_analyzer
except ImportError:
    trend_analyzer = None
from ..agents.format_generator import format_generator

# Set up logging
logger = logging.getLogger(__name__)


class AsyncEventSourcedMemeService:
    """Async event-sourced service for meme operations."""
    
    def __init__(self):
        """Initialize the async meme service."""
        logger.info("Initializing async event-sourced meme service")
    
    async def get_memes(
        self,
        status: Optional[str] = None,
        topic: Optional[str] = None,
        format: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get memes using read model projections.
        
        Args:
            status: Filter by status (generating, completed, failed, etc.)
            topic: Filter by topic
            format: Filter by format
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of memes from projections
        """
        logger.info(f"Getting memes: status={status}, topic={topic}, format={format}, limit={limit}, offset={offset}")
        return await list_meme_projections(status, topic, format, limit, offset)
    
    async def get_meme(self, meme_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a meme by ID using read model projection.
        
        Args:
            meme_id: ID of the meme
            
        Returns:
            Meme data if found, None otherwise
        """
        logger.info(f"Getting meme with ID: {meme_id}")
        
        try:
            meme_uuid = UUID(meme_id)
            result = await get_meme_projection(meme_uuid)
            
            if result:
                # Emit view event for analytics
                from ..cqrs.events import EventMetadata
                
                metadata = EventMetadata(
                    aggregate_id=meme_uuid,
                    aggregate_type="Meme"
                )
                
                view_event = MemeViewed(
                    meme_id=meme_uuid, 
                    view_source="api"
                )
                view_event.metadata = metadata
                await publish_event(view_event)
            
            return result
            
        except ValueError:
            logger.error(f"Invalid meme ID format: {meme_id}")
            return None
    
    async def create_meme(
        self,
        topic: str,
        format_id: str,
        style: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new meme using async event sourcing and actor system.
        
        Args:
            topic: Topic for the meme
            format_id: Format ID for the meme
            style: Optional style parameters
            parameters: Additional generation parameters
            user_id: User ID for tracking
            
        Returns:
            Created meme data with request tracking info
        """
        logger.info(f"Creating meme with topic: {topic}, format: {format_id}")
        
        # Create command with proper metadata
        from ..cqrs.commands.base import CommandMetadata
        
        metadata = CommandMetadata(
            user_id=user_id,
            correlation_id=uuid4()
        )
        
        command = GenerateMemeCommand(
            topic=topic,
            format=format_id,
            style=style,
            parameters=parameters or {}
        )
        command.metadata = metadata
        
        # Execute command through CQRS
        result = await handle_command(command)
        
        if result.success:
            meme_id = result.value
            logger.info(f"Meme generation started with ID: {meme_id}")
            
            return {
                "meme_id": str(meme_id),
                "request_id": str(command.metadata.command_id),
                "status": "generating",
                "topic": topic,
                "format": format_id,
                "correlation_id": str(command.metadata.correlation_id),
                "message": "Meme generation started. Use the request_id to track progress."
            }
        else:
            logger.error(f"Meme generation failed: {result.error}")
            raise Exception(f"Meme generation failed: {result.error}")
    
    async def get_generation_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a meme generation request.
        
        Args:
            request_id: Request ID from create_meme
            
        Returns:
            Pipeline status information
        """
        logger.info(f"Getting generation status for request: {request_id}")
        
        try:
            request_uuid = UUID(request_id)
            return await get_pipeline_status(request_uuid)
        except ValueError:
            logger.error(f"Invalid request ID format: {request_id}")
            return None
    
    async def score_meme(self, meme_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Score an existing meme.
        
        Args:
            meme_id: ID of the meme to score
            user_id: User ID for tracking
            
        Returns:
            Scoring result
        """
        logger.info(f"Scoring meme with ID: {meme_id}")
        
        try:
            meme_uuid = UUID(meme_id)
            
            # Create command with proper metadata
            from ..cqrs.commands.base import CommandMetadata
            
            metadata = CommandMetadata(
                user_id=user_id,
                correlation_id=uuid4()
            )
            
            command = ScoreMemeCommand(
                meme_id=meme_uuid
            )
            command.metadata = metadata
            
            # Execute command
            result = await handle_command(command)
            
            if result.success:
                return {
                    "meme_id": meme_id,
                    "score": result.value,
                    "message": "Meme scored successfully"
                }
            else:
                raise Exception(f"Scoring failed: {result.error}")
                
        except ValueError:
            logger.error(f"Invalid meme ID format: {meme_id}")
            raise Exception("Invalid meme ID format")
    
    async def delete_meme(
        self,
        meme_id: str,
        reason: str,
        user_id: Optional[str] = None,
        soft_delete: bool = True
    ) -> Dict[str, Any]:
        """
        Delete a meme.
        
        Args:
            meme_id: ID of the meme to delete
            reason: Reason for deletion
            user_id: User ID for tracking
            soft_delete: Whether to soft delete (default) or hard delete
            
        Returns:
            Deletion result
        """
        logger.info(f"Deleting meme with ID: {meme_id}, reason: {reason}")
        
        try:
            meme_uuid = UUID(meme_id)
            
            # Create command with proper metadata
            from ..cqrs.commands.base import CommandMetadata
            
            metadata = CommandMetadata(
                user_id=user_id,
                correlation_id=uuid4()
            )
            
            command = DeleteMemeCommand(
                meme_id=meme_uuid,
                deletion_reason=reason,
                soft_delete=soft_delete
            )
            command.metadata = metadata
            
            # Execute command
            result = await handle_command(command)
            
            if result.success:
                return {
                    "meme_id": meme_id,
                    "deleted": True,
                    "soft_delete": soft_delete,
                    "message": "Meme deleted successfully"
                }
            else:
                raise Exception(f"Deletion failed: {result.error}")
                
        except ValueError:
            logger.error(f"Invalid meme ID format: {meme_id}")
            raise Exception("Invalid meme ID format")
    
    async def share_meme(
        self,
        meme_id: str,
        platform: str,
        method: str = "api",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a meme share event.
        
        Args:
            meme_id: ID of the meme being shared
            platform: Platform where it's being shared
            method: Share method
            user_id: User ID for tracking
            
        Returns:
            Share confirmation
        """
        logger.info(f"Recording share for meme {meme_id} on {platform}")
        
        try:
            meme_uuid = UUID(meme_id)
            
            # Emit share event
            from ..cqrs.events import EventMetadata
            
            metadata = EventMetadata(
                aggregate_id=meme_uuid,
                aggregate_type="Meme",
                user_id=user_id
            )
            
            share_event = MemeShared(
                meme_id=meme_uuid,
                sharer_id=user_id,
                share_platform=platform,
                share_method=method
            )
            share_event.metadata = metadata
            
            await publish_event(share_event)
            
            return {
                "meme_id": meme_id,
                "platform": platform,
                "shared": True,
                "message": "Share recorded successfully"
            }
            
        except ValueError:
            logger.error(f"Invalid meme ID format: {meme_id}")
            raise Exception("Invalid meme ID format")
    
    async def get_metrics(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Metrics data
        """
        from datetime import datetime
        
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise Exception("Invalid date format. Use YYYY-MM-DD")
        else:
            target_date = datetime.utcnow()
        
        logger.info(f"Getting metrics for date: {target_date.strftime('%Y-%m-%d')}")
        return await get_daily_metrics(target_date)
    
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


# Legacy service for backward compatibility
class MemeService(AsyncEventSourcedMemeService):
    """Legacy wrapper maintaining backward compatibility."""
    pass


# Create singleton instances
async_meme_service = AsyncEventSourcedMemeService()
meme_service = MemeService()  # Maintains backward compatibility 