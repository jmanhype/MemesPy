"""CQRS module initialization and integration.

This module provides the main integration point for the CQRS event sourcing system,
including event store, event bus, projections, and command handlers.
"""

import asyncio
import logging
from typing import Optional

from ..config.config import settings

# Import core components
from .event_store import initialize_event_store, get_event_store, shutdown_event_store
from .event_bus import get_event_bus, shutdown_event_bus
from .projections import initialize_projections, rebuild_projections_from_events
# from .command_handlers import initialize_command_handlers  # Temporarily disabled due to actor message issues

logger = logging.getLogger(__name__)


class CQRSSystem:
    """Main CQRS system coordinator."""
    
    def __init__(self):
        self._initialized = False
        self._event_store = None
        self._event_bus = None
        self._actor_system = None
    
    async def initialize(self, actor_system=None) -> None:
        """Initialize the CQRS system."""
        if self._initialized:
            logger.warning("CQRS system already initialized")
            return
        
        logger.info("Initializing CQRS event sourcing system")
        
        try:
            # Initialize event store
            if settings.enable_event_sourcing:
                logger.info("Initializing event store")
                self._event_store = await initialize_event_store(
                    settings.event_store_connection_string
                )
            
            # Initialize event bus
            logger.info("Initializing event bus")
            self._event_bus = await get_event_bus()
            
            # Initialize projections
            if settings.enable_projections:
                logger.info("Initializing projections")
                await initialize_projections()
                
                # Rebuild projections if configured
                if settings.projection_rebuild_on_startup:
                    logger.info("Rebuilding projections from events")
                    await rebuild_projections_from_events()
            
            # Initialize command handlers with actor system
            if actor_system and settings.enable_actor_system:
                logger.info("Actor system provided but command handlers disabled due to import issues")
                self._actor_system = actor_system
                # await initialize_command_handlers(actor_system)  # Temporarily disabled
            
            self._initialized = True
            logger.info("CQRS system initialization completed successfully")
            
        except Exception as e:
            logger.error(f"CQRS system initialization failed: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the CQRS system."""
        if not self._initialized:
            return
        
        logger.info("Shutting down CQRS system")
        
        try:
            # Shutdown event bus first to stop processing
            if self._event_bus:
                await shutdown_event_bus()
                self._event_bus = None
            
            # Shutdown event store
            if self._event_store:
                await shutdown_event_store()
                self._event_store = None
            
            self._actor_system = None
            self._initialized = False
            
            logger.info("CQRS system shutdown completed")
            
        except Exception as e:
            logger.error(f"CQRS system shutdown failed: {e}", exc_info=True)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the CQRS system is initialized."""
        return self._initialized
    
    async def get_event_store(self):
        """Get the event store instance."""
        if not self._initialized:
            raise RuntimeError("CQRS system not initialized")
        return await get_event_store()
    
    async def get_event_bus(self):
        """Get the event bus instance."""
        if not self._initialized:
            raise RuntimeError("CQRS system not initialized")
        return await get_event_bus()


# Global CQRS system instance
_cqrs_system: Optional[CQRSSystem] = None


async def initialize_cqrs(actor_system=None) -> CQRSSystem:
    """Initialize the global CQRS system."""
    global _cqrs_system
    
    if _cqrs_system is None:
        _cqrs_system = CQRSSystem()
    
    await _cqrs_system.initialize(actor_system)
    return _cqrs_system


async def get_cqrs_system() -> CQRSSystem:
    """Get the global CQRS system instance."""
    global _cqrs_system
    
    if _cqrs_system is None or not _cqrs_system.is_initialized:
        raise RuntimeError("CQRS system not initialized. Call initialize_cqrs() first.")
    
    return _cqrs_system


async def shutdown_cqrs() -> None:
    """Shutdown the global CQRS system."""
    global _cqrs_system
    
    if _cqrs_system:
        await _cqrs_system.shutdown()
        _cqrs_system = None


# Convenience functions for common operations
async def is_cqrs_available() -> bool:
    """Check if CQRS system is available and initialized."""
    try:
        system = await get_cqrs_system()
        return system.is_initialized
    except RuntimeError:
        return False


async def ensure_cqrs_initialized(actor_system=None) -> None:
    """Ensure CQRS system is initialized, initializing if needed."""
    try:
        await get_cqrs_system()
    except RuntimeError:
        logger.info("CQRS system not initialized, initializing now")
        await initialize_cqrs(actor_system)


# Export key components
__all__ = [
    'CQRSSystem',
    'initialize_cqrs',
    'get_cqrs_system',
    'shutdown_cqrs',
    'is_cqrs_available',
    'ensure_cqrs_initialized'
]