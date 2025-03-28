"""Minimal FastAPI app to test mock implementations."""

import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import mock implementations
from src.dspy_meme_gen.agents.meme_generator import meme_generator
from src.dspy_meme_gen.agents.trend_analyzer import trend_analyzer
from src.dspy_meme_gen.agents.format_generator import format_generator

# Create FastAPI app
app = FastAPI(title="DSPy Mock API", description="A minimal API for testing mock DSPy implementations")


@app.get("/")
async def root():
    """Redirect to the docs."""
    return {"message": "Welcome to the DSPy Mock API", "docs_url": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/formats")
async def get_formats():
    """Get all available formats."""
    formats = await format_generator.get_formats()
    return {"items": formats, "count": len(formats)}


@app.get("/api/formats/{format_id}")
async def get_format(format_id: str):
    """Get a specific format by ID."""
    format_data = await format_generator.get_format(format_id)
    if not format_data:
        return JSONResponse(status_code=404, content={"error": f"Format with ID {format_id} not found"})
    return format_data


@app.get("/api/trends")
async def get_trends():
    """Get all trending topics."""
    trends = await trend_analyzer.get_trending_topics()
    return {"items": trends, "count": len(trends)}


@app.get("/api/trends/{trend_id}")
async def get_trend(trend_id: str):
    """Get a specific trend by ID."""
    trend = await trend_analyzer.get_trending_topic(trend_id)
    if not trend:
        return JSONResponse(status_code=404, content={"error": f"Trend with ID {trend_id} not found"})
    return trend


@app.post("/api/memes")
async def create_meme(topic: str, format_id: str):
    """Create a new meme."""
    meme = await meme_generator.generate_meme(topic, format_id)
    return meme


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 