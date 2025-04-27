"""Database models for memes."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MemeDB(Base):
    """
    Database model for memes.
    
    Attributes:
        id: Unique identifier for the meme
        topic: The topic of the meme
        format: The format of the meme
        text: The text content of the meme
        image_url: URL to the generated meme image
        created_at: Timestamp when the meme was created
        score: Quality score of the meme (0.0 to 1.0)
    """
    
    __tablename__ = "memes"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String, nullable=False, index=True)
    format = Column(String, nullable=False)
    text = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # score = Column(Float, nullable=False) # Removed score column 