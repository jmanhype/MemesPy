"""Pydantic schemas for format-related operations."""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class FormatResponse(BaseModel):
    """Schema for format response."""
    
    id: str = Field(..., description="Unique identifier for the format")
    name: str = Field(..., description="Name of the meme format")
    description: str = Field(..., description="Description of the meme format")
    example_url: str = Field(..., description="URL to an example of this meme format")
    popularity: float = Field(..., description="Popularity score of the format", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "standard",
                "name": "Standard",
                "description": "Standard meme format with image and top/bottom text",
                "example_url": "https://example.com/standard.jpg",
                "popularity": 0.9
            }
        }


class FormatListResponse(BaseModel):
    """Schema for list of formats response."""
    
    items: List[FormatResponse] = Field(..., description="List of formats")
    total: int = Field(..., description="Total number of formats")
    limit: int = Field(..., description="Maximum number of formats per page")
    offset: int = Field(..., description="Offset for pagination")
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "standard",
                        "name": "Standard",
                        "description": "Standard meme format with image and top/bottom text",
                        "example_url": "https://example.com/standard.jpg",
                        "popularity": 0.9
                    },
                    {
                        "id": "modern",
                        "name": "Modern",
                        "description": "Modern meme format with image and integrated text",
                        "example_url": "https://example.com/modern.jpg",
                        "popularity": 0.8
                    }
                ],
                "total": 5,
                "limit": 10,
                "offset": 0
            }
        } 