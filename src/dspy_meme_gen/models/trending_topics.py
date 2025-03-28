"""Models for trending topics."""

from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func

from .base import Base

class TrendingTopic(Base):
    """Model for trending topics."""
    
    __tablename__ = "trending_topics"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    topic = Column(String(255), nullable=False)
    source = Column(String(50))
    relevance_score = Column(Float, default=0.0)
    timestamp = Column(DateTime, server_default=func.now())
    trend_metadata = Column(JSON)
    
    def __init__(
        self,
        topic: str,
        source: Optional[str] = None,
        relevance_score: float = 0.0,
        trend_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a TrendingTopic.
        
        Args:
            topic: The trending topic
            source: Source of the trend (e.g., "twitter", "reddit")
            relevance_score: Relevance score for ranking
            trend_metadata: Additional metadata about the trend
        """
        self.topic = topic
        self.source = source
        self.relevance_score = relevance_score
        self.trend_metadata = trend_metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        return {
            "id": self.id,
            "topic": self.topic,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "trend_metadata": self.trend_metadata
        } 