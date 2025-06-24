"""Pydantic schemas for trend-related operations."""

from pydantic import BaseModel, Field
from typing import List, Optional


class TrendResponse(BaseModel):
    """Schema for trend response."""

    id: str = Field(..., description="Unique identifier for the trend")
    name: str = Field(..., description="Name of the trending topic")
    description: str = Field(..., description="Description of the trending topic")
    popularity: float = Field(..., description="Popularity score of the trend", ge=0.0, le=1.0)
    suggested_formats: List[str] = Field(
        ..., description="List of suggested meme formats for this trend"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "trend123",
                "name": "AI Ethics",
                "description": "Memes about AI and ethical considerations",
                "popularity": 0.95,
                "suggested_formats": ["standard", "modern"],
            }
        }


class TrendListResponse(BaseModel):
    """Schema for list of trends response."""

    items: List[TrendResponse] = Field(..., description="List of trends")
    total: int = Field(..., description="Total number of trends")
    limit: int = Field(..., description="Maximum number of trends per page")
    offset: int = Field(..., description="Offset for pagination")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "trend123",
                        "name": "AI Ethics",
                        "description": "Memes about AI and ethical considerations",
                        "popularity": 0.95,
                        "suggested_formats": ["standard", "modern"],
                    },
                    {
                        "id": "trend456",
                        "name": "Python vs JavaScript",
                        "description": "Programming language rivalry memes",
                        "popularity": 0.85,
                        "suggested_formats": ["comparison", "standard"],
                    },
                ],
                "total": 42,
                "limit": 10,
                "offset": 0,
            }
        }
