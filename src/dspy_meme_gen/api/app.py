"""FastAPI application module."""
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app(debug: bool = False) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        debug: Enable debug mode.
        
    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(
        title="DSPy Meme Generator",
        description="A sophisticated meme generation pipeline using DSPy",
        version="0.1.0",
        debug=debug,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routers
    from dspy_meme_gen.api.health import router as health_router
    app.include_router(health_router)
    
    return app 