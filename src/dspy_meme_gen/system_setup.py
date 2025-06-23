"""System setup for async event sourcing and actor integration."""

import asyncio
import logging
from typing import Optional

from .config.config import settings
from .actors.core import ActorSystem
from .actors.text_generator_actor import TextGeneratorActor
from .actors.image_generator_actor import ImageGeneratorActor
from .actors.meme_generator_actor import MemeGeneratorActor
from .cqrs import (
    initialize_event_store, get_event_store, shutdown_event_store,
    get_event_bus, shutdown_event_bus,
    initialize_command_handlers, initialize_projections
)

logger = logging.getLogger(__name__)


class AsyncEventSourcedSystem:
    """Main system class for managing async event sourcing and actors."""
    
    def __init__(self):
        self.actor_system: Optional[ActorSystem] = None
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize the complete async event sourcing system."""
        if self.initialized:
            logger.warning("System already initialized")
            return
            
        logger.info("Initializing async event sourcing system...")
        
        try:
            # 1. Initialize event store
            logger.info("Initializing event store...")
            await initialize_event_store(settings.database_url)
            
            # 2. Initialize event bus
            logger.info("Initializing event bus...")
            await get_event_bus()  # This creates and starts the event bus
            
            # 3. Initialize projections
            logger.info("Initializing projections...")
            await initialize_projections()
            
            # 4. Initialize actor system
            logger.info("Initializing actor system...")
            self.actor_system = ActorSystem("meme-generation")
            await self.actor_system.start()
            
            # 5. Spawn core actors
            await self._spawn_core_actors()
            
            # 6. Initialize command handlers with actors
            logger.info("Initializing command handlers...")
            await initialize_command_handlers(self.actor_system)
            
            self.initialized = True
            logger.info("Async event sourcing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def _spawn_core_actors(self) -> None:
        """Spawn the core actors needed for meme generation."""
        # Text generator actor
        text_actor = TextGeneratorActor(
            name="text_generator",
            openai_api_key=settings.openai_api_key,
            model_name=settings.dspy_model
        )
        await self.actor_system.register_actor(text_actor)
        
        # Image generator actor
        image_actor = ImageGeneratorActor(
            name="image_generator",
            openai_api_key=settings.openai_api_key
        )
        await self.actor_system.register_actor(image_actor)
        
        # Meme generator orchestrator actor
        meme_actor = MemeGeneratorActor(
            name="meme_orchestrator",
            text_generator_ref=self.actor_system.get_actor("text_generator"),
            image_generator_ref=self.actor_system.get_actor("image_generator")
        )
        await self.actor_system.register_actor(meme_actor)
        
        # Quality scorer actor (if we have one)
        try:
            from .actors.quality_scorer_actor import QualityScorerActor
            scorer_actor = QualityScorerActor(
                name="quality_scorer",
                model_name=settings.dspy_model
            )
            await self.actor_system.register_actor(scorer_actor)
        except ImportError:
            logger.warning("Quality scorer actor not available")
        
        # Verifier actor (if we have one)
        try:
            from .actors.verifier_actor import VerifierActor
            verifier_actor = VerifierActor(
                name="verifier", 
                model_name=settings.dspy_model
            )
            await self.actor_system.register_actor(verifier_actor)
        except ImportError:
            logger.warning("Verifier actor not available")
        
        logger.info("Core actors spawned successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        if not self.initialized:
            return
            
        logger.info("Shutting down async event sourcing system...")
        
        # 1. Shutdown actor system
        if self.actor_system:
            await self.actor_system.stop()
            self.actor_system = None
        
        # 2. Shutdown event bus
        await shutdown_event_bus()
        
        # 3. Shutdown event store
        await shutdown_event_store()
        
        self.initialized = False
        logger.info("System shutdown complete")
    
    def get_actor_system(self) -> Optional[ActorSystem]:
        """Get the actor system instance."""
        return self.actor_system
    
    def is_initialized(self) -> bool:
        """Check if system is initialized."""
        return self.initialized


# Global system instance
_system: Optional[AsyncEventSourcedSystem] = None


async def initialize_system() -> AsyncEventSourcedSystem:
    """Initialize the global system."""
    global _system
    if _system is None:
        _system = AsyncEventSourcedSystem()
        await _system.initialize()
    return _system


async def get_system() -> AsyncEventSourcedSystem:
    """Get the global system instance."""
    global _system
    if _system is None or not _system.is_initialized():
        return await initialize_system()
    return _system


async def shutdown_system() -> None:
    """Shutdown the global system."""
    global _system
    if _system:
        await _system.shutdown()
        _system = None


# Context manager for system lifecycle
class SystemManager:
    """Context manager for system lifecycle."""
    
    def __init__(self):
        self.system: Optional[AsyncEventSourcedSystem] = None
    
    async def __aenter__(self) -> AsyncEventSourcedSystem:
        self.system = await get_system()
        return self.system
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.system:
            await self.system.shutdown()


# Health check functions
async def health_check() -> dict:
    """Perform system health check."""
    try:
        system = await get_system()
        
        health = {
            "system_initialized": system.is_initialized(),
            "actor_system": False,
            "event_store": False,
            "event_bus": False,
            "actors": {}
        }
        
        # Check actor system
        if system.actor_system:
            health["actor_system"] = system.actor_system._running
            
            # Check individual actors
            for name, actor in system.actor_system.actors.items():
                health["actors"][name] = actor.running
        
        # Check event store
        try:
            event_store = await get_event_store()
            # Try a simple query to verify connection
            await event_store.get_events_by_type("health_check", limit=1)
            health["event_store"] = True
        except:
            health["event_store"] = False
        
        # Check event bus
        try:
            event_bus = await get_event_bus()
            health["event_bus"] = event_bus._running
        except:
            health["event_bus"] = False
        
        health["overall"] = all([
            health["system_initialized"],
            health["actor_system"],
            health["event_store"],
            health["event_bus"]
        ])
        
        return health
        
    except Exception as e:
        return {
            "overall": False,
            "error": str(e),
            "system_initialized": False,
            "actor_system": False,
            "event_store": False,
            "event_bus": False,
            "actors": {}
        }


# Initialization helper for FastAPI startup
async def startup_system():
    """Startup function for FastAPI app."""
    try:
        await initialize_system()
        logger.info("System startup completed")
    except Exception as e:
        logger.error(f"System startup failed: {e}", exc_info=True)
        raise


# Shutdown helper for FastAPI shutdown
async def shutdown_system_handler():
    """Shutdown function for FastAPI app."""
    try:
        await shutdown_system()
        logger.info("System shutdown completed")
    except Exception as e:
        logger.error(f"System shutdown failed: {e}", exc_info=True)