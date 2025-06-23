"""Pydantic schemas for meme models with async event sourcing support."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class MemeGenerationRequest(BaseModel):
    """
    Schema for meme generation request.
    
    Attributes:
        topic: The topic for the meme
        format: The meme format to use
        context: Optional context for generation
        style: Optional style parameters
        parameters: Additional generation parameters
    """
    
    topic: str = Field(..., description="The topic for the meme", min_length=1, max_length=100)
    format: str = Field(..., description="The meme format to use", min_length=1, max_length=50)
    context: Optional[str] = Field(None, description="Optional context for meme generation")
    style: Optional[str] = Field(None, description="Optional style parameters")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional generation parameters")


class MemeResponse(BaseModel):
    """
    Schema for meme response with event sourcing support.
    
    Attributes:
        id: Unique identifier for the meme
        topic: The topic of the meme
        format: The format of the meme
        text: The text content of the meme
        image_url: URL to the generated meme image
        created_at: Timestamp when the meme was created
        status: Current status of the meme (for async generation)
        request_id: Request ID for tracking generation progress
        message: Status message
        scores: Quality scores
        metadata: Additional metadata
    """
    
    id: str = Field(..., description="Unique identifier for the meme")
    topic: str = Field(..., description="The topic for the meme")
    format: str = Field(..., description="The meme format used")
    text: str = Field(..., description="The text content of the meme")
    image_url: str = Field(..., description="URL to the generated meme image")
    created_at: str = Field(..., description="Creation timestamp")
    
    # New event sourcing fields
    status: Optional[str] = Field(None, description="Current status of the meme")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    message: Optional[str] = Field(None, description="Status message")
    scores: Optional[Dict[str, float]] = Field(None, description="Quality scores")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("created_at", pre=True)
    def parse_datetime(cls, value):
        """Convert datetime to string if needed."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class MemeListResponse(BaseModel):
    """
    Schema for list of memes response.
    
    Attributes:
        items: List of memes
        total: Total number of memes
        limit: Maximum number of memes per page
        offset: Offset for pagination
    """
    
    items: List[MemeResponse] = Field(..., description="List of memes")
    total: int = Field(..., description="Total number of memes")
    limit: int = Field(..., description="Maximum number of memes per page")
    offset: int = Field(..., description="Offset for pagination")