"""Database models for content guidelines."""

from typing import Dict, Any, Optional, List
from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum
from pydantic import BaseModel, Field

from .base import Base


class SeverityLevel(str, PyEnum):
    """Enumeration for content severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GuidelineCategory(Base):
    """Model for guideline categories (e.g., language, content, cultural)."""
    __tablename__ = "guideline_categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(String(255))
    
    # Relationship to guidelines
    guidelines = relationship("ContentGuideline", back_populates="category")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the category to a dictionary.
        
        Returns:
            Dictionary representation of the category
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }


class ContentGuideline(BaseModel):
    """Content guidelines for meme generation.
    
    Attributes:
        allowed_topics: List of allowed topics
        forbidden_topics: List of forbidden topics
        min_age_rating: Minimum age rating
        max_age_rating: Maximum age rating
        language_filter: Language filter level
        violence_filter: Violence filter level
        adult_content_filter: Adult content filter level
    """
    
    allowed_topics: Optional[List[str]] = Field(default_factory=list)
    forbidden_topics: Optional[List[str]] = Field(default_factory=list)
    min_age_rating: Optional[int] = Field(default=0, ge=0, le=18)
    max_age_rating: Optional[int] = Field(default=18, ge=0, le=18)
    language_filter: str = Field(default="moderate", pattern="^(strict|moderate|none)$")
    violence_filter: str = Field(default="moderate", pattern="^(strict|moderate|none)$")
    adult_content_filter: str = Field(default="strict", pattern="^(strict|moderate|none)$")
    
    def is_topic_allowed(self, topic: str) -> bool:
        """Check if a topic is allowed.
        
        Args:
            topic: Topic to check
            
        Returns:
            True if topic is allowed, False otherwise
        """
        if self.forbidden_topics and topic.lower() in [t.lower() for t in self.forbidden_topics]:
            return False
            
        if self.allowed_topics and topic.lower() not in [t.lower() for t in self.allowed_topics]:
            return False
            
        return True
        
    def is_age_appropriate(self, age_rating: int) -> bool:
        """Check if content is age appropriate.
        
        Args:
            age_rating: Age rating to check
            
        Returns:
            True if age appropriate, False otherwise
        """
        return self.min_age_rating <= age_rating <= self.max_age_rating
        
    def get_filter_levels(self) -> dict:
        """Get filter levels.
        
        Returns:
            Dictionary of filter levels
        """
        return {
            "language": self.language_filter,
            "violence": self.violence_filter,
            "adult_content": self.adult_content_filter
        }


class GuidelineRepository:
    """Repository for managing content guidelines in the database."""

    def __init__(self, session_factory) -> None:
        """
        Initialize the repository.
        
        Args:
            session_factory: SQLAlchemy session factory
        """
        self.session_factory = session_factory

    def get_all_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all guidelines organized by category.
        
        Returns:
            Dictionary of guidelines organized by category
        """
        with self.session_factory() as session:
            categories = session.query(GuidelineCategory).all()
            
            result = {}
            for category in categories:
                result[category.name] = {
                    guideline.name: {
                        "severity": guideline.severity.value,
                        "description": guideline.description,
                        **(guideline.guideline_metadata or {})
                    }
                    for guideline in category.guidelines
                }
            
            return result

    def initialize_default_guidelines(self) -> None:
        """Initialize the database with default content guidelines."""
        default_categories = [
            {
                "name": "language",
                "description": "Guidelines for language usage",
                "guidelines": [
                    {
                        "name": "profanity",
                        "description": "Use of explicit or offensive language",
                        "severity": SeverityLevel.HIGH
                    },
                    {
                        "name": "slurs",
                        "description": "Use of discriminatory or hateful terms",
                        "severity": SeverityLevel.HIGH
                    },
                    {
                        "name": "tone",
                        "description": "Overly aggressive or confrontational tone",
                        "severity": SeverityLevel.MEDIUM
                    }
                ]
            },
            {
                "name": "content",
                "description": "Guidelines for content themes",
                "guidelines": [
                    {
                        "name": "violence",
                        "description": "Depiction or suggestion of violence",
                        "severity": SeverityLevel.HIGH
                    },
                    {
                        "name": "discrimination",
                        "description": "Content that discriminates against groups",
                        "severity": SeverityLevel.HIGH
                    },
                    {
                        "name": "sensitive_topics",
                        "description": "References to sensitive social/political issues",
                        "severity": SeverityLevel.MEDIUM
                    }
                ]
            },
            {
                "name": "cultural",
                "description": "Guidelines for cultural sensitivity",
                "guidelines": [
                    {
                        "name": "stereotypes",
                        "description": "Use of cultural or ethnic stereotypes",
                        "severity": SeverityLevel.HIGH
                    },
                    {
                        "name": "appropriation",
                        "description": "Inappropriate use of cultural elements",
                        "severity": SeverityLevel.MEDIUM
                    },
                    {
                        "name": "context",
                        "description": "Lack of cultural context or sensitivity",
                        "severity": SeverityLevel.MEDIUM
                    }
                ]
            },
            {
                "name": "professional",
                "description": "Guidelines for professional context",
                "guidelines": [
                    {
                        "name": "workplace",
                        "description": "Inappropriate for professional settings",
                        "severity": SeverityLevel.MEDIUM
                    },
                    {
                        "name": "brand_safety",
                        "description": "Potential brand reputation risks",
                        "severity": SeverityLevel.MEDIUM
                    }
                ]
            }
        ]

        with self.session_factory() as session:
            for cat_data in default_categories:
                # Create category if it doesn't exist
                category = session.query(GuidelineCategory).filter_by(
                    name=cat_data["name"]
                ).first()
                
                if not category:
                    category = GuidelineCategory(
                        name=cat_data["name"],
                        description=cat_data["description"]
                    )
                    session.add(category)
                    session.flush()  # Get the ID
                
                # Add guidelines
                for guide_data in cat_data["guidelines"]:
                    guideline = session.query(ContentGuideline).filter_by(
                        category_id=category.id,
                        name=guide_data["name"]
                    ).first()
                    
                    if not guideline:
                        guideline = ContentGuideline(
                            category_id=category.id,
                            name=guide_data["name"],
                            description=guide_data["description"],
                            severity=guide_data["severity"]
                        )
                        session.add(guideline)
            
            session.commit() 