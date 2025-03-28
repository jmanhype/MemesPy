"""Models for meme-related data."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemeFormat(BaseModel):
    """Model for meme format data."""
    
    id: str
    name: str
    description: str
    popularity: float = Field(ge=0.0, le=1.0)


class TrendingTopic(BaseModel):
    """Model for trending topic data."""
    
    id: str
    name: str
    description: str
    popularity: float = Field(ge=0.0, le=1.0)
    suggested_formats: List[str]


class MemeData(BaseModel):
    """Model for meme data."""
    
    id: str
    topic: str
    format: str
    text: str
    image_url: str
    created_at: str
    score: float = Field(ge=0.0, le=1.0)


class CreateMemeRequest(BaseModel):
    """Model for create meme request."""
    
    topic: str
    format: str
    style: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None 