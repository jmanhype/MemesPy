"""Request models for the API."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class CreateMemeRequest(BaseModel):
    """Request model for creating a new meme."""

    topic: str = Field(..., description="Topic for the meme")
    format: str = Field(..., description="Format ID for the meme")

    class Config:
        """Configuration for the model."""

        json_schema_extra = {"example": {"topic": "Python Programming", "format": "standard"}}
