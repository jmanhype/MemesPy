"""FastAPI main application."""

import logging
import dspy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles

from ..config.config import settings
from .routers import health, memes, analytics, privacy
from ..models.db_models.memes import Base
from ..models.db_models.metadata import Base as MetadataBase
from ..models.db_models.privacy_metadata import Base as PrivacyBase
from .middleware.privacy_middleware import PrivacyMiddleware, ConsentEnforcementMiddleware
from ..models.connection import db_manager

# Import dependency for shutdown event
from .dependencies import close_connections

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# Create database tables
# NOTE: Commented out for async SQLite - tables will be created on startup
# Base.metadata.create_all(bind=db_manager.sync_engine)
# MetadataBase.metadata.create_all(bind=db_manager.sync_engine)
# PrivacyBase.metadata.create_all(bind=db_manager.sync_engine)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="A meme generation service using DSPy",
    version=settings.app_version,
    docs_url="/docs",
)

# --- Mount Static Files Directory ---
# This will serve files from the 'static' directory at the '/static' URL path
app.mount("/static", StaticFiles(directory="static"), name="static")
# --- End Static Files Mounting ---

# Add privacy middleware FIRST (before CORS)
app.add_middleware(PrivacyMiddleware)
app.add_middleware(ConsentEnforcementMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(memes.router, prefix="/api/v1/memes", tags=["memes"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(privacy.router, prefix="/api", tags=["privacy"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup, including DSPy configuration and async event sourcing."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.app_env}")

    # Start privacy tasks
    from ..tasks.privacy_tasks import start_privacy_tasks

    start_privacy_tasks()

    # --- Configure DSPy ---
    if settings.openai_api_key and settings.dspy_model:
        try:
            logger.info(f"Configuring DSPy with model: {settings.dspy_model}...")
            # Using dspy.OpenAI as an example, adjust if using Anthropic, Cohere, etc.
            # Make sure the model name format matches what dspy.OpenAI expects
            # e.g., might just be 'gpt-3.5-turbo' or need 'openai/gpt-3.5-turbo'
            # Depending on the LM class used. Let's assume direct model name for now.
            # lm = dspy.OpenAI(model=settings.dspy_model, api_key=settings.openai_api_key)

            # Corrected way using dspy.LM with provider prefix:
            lm = dspy.LM(model=f"openai/{settings.dspy_model}", api_key=settings.openai_api_key)

            dspy.configure(lm=lm)
            logger.info("DSPy configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}", exc_info=True)
            # Depending on requirements, you might want to raise an exception here
            # or allow the app to start without DSPy functionality (fallback will be used).
    else:
        logger.warning(
            "DSPy configuration skipped: OpenAI API key or DSPy model name missing in settings."
        )
    # --- End DSPy Configuration ---

    # --- Initialize Async Event Sourcing System ---
    try:
        logger.info("Initializing async event sourcing system...")
        from ..system_setup import startup_system

        await startup_system()
        logger.info("Async event sourcing system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize async event sourcing system: {e}", exc_info=True)
        # Note: You may want to decide if the app should continue without event sourcing
        # For now, we'll log the error but continue startup
    # --- End Event Sourcing Initialization ---


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info(f"Shutting down {settings.app_name}")

    # --- Shutdown Async Event Sourcing System ---
    try:
        logger.info("Shutting down async event sourcing system...")
        from ..system_setup import shutdown_system_handler

        await shutdown_system_handler()
        logger.info("Async event sourcing system shutdown completed")
    except Exception as e:
        logger.error(f"Failed to shutdown async event sourcing system: {e}", exc_info=True)
    # --- End Event Sourcing Shutdown ---

    # Stop privacy tasks
    from ..tasks.privacy_tasks import stop_privacy_tasks

    stop_privacy_tasks()

    # Close cache connections
    await close_connections()
