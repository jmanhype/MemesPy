"""Simple FastAPI application with database support."""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "sqlite:///memes.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class MemeDB(Base):
    """Database model for memes."""
    
    __tablename__ = "memes"
    
    id = Column(String, primary_key=True, index=True)
    topic = Column(String, nullable=False, index=True)
    format = Column(String, nullable=False)
    text = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    score = Column(Float, nullable=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency for database session
def get_db():
    """Get database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI app
app = FastAPI(
    title="DSPy Meme Generator",
    description="A meme generation service with database storage",
    version="0.1.0"
)

# API routes
@app.get("/api/health")
def health_check():
    """Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "database": "connected"
    }

@app.get("/api/v1/memes/", response_model=List[Dict[str, Any]])
def list_memes(db: Session = Depends(get_db)):
    """List all memes.
    
    Args:
        db: Database session
        
    Returns:
        List of memes
    """
    memes = db.query(MemeDB).all()
    return [
        {
            "id": meme.id,
            "topic": meme.topic,
            "format": meme.format,
            "text": meme.text,
            "image_url": meme.image_url,
            "created_at": meme.created_at.isoformat(),
            "score": meme.score
        }
        for meme in memes
    ]

@app.post("/api/v1/memes/", status_code=status.HTTP_201_CREATED)
def create_meme(meme: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new meme.
    
    Args:
        meme: Meme data
        db: Database session
        
    Returns:
        Created meme
    """
    # Generate example meme
    meme_id = str(uuid.uuid4())
    db_meme = MemeDB(
        id=meme_id,
        topic=meme.get("topic", "Example"),
        format=meme.get("format", "standard"),
        text="Sample meme text about " + meme.get("topic", "programming"),
        image_url="https://example.com/sample.jpg",
        created_at=datetime.utcnow(),
        score=0.95
    )
    
    # Store in database
    db.add(db_meme)
    db.commit()
    db.refresh(db_meme)
    
    # Return response
    return {
        "id": db_meme.id,
        "topic": db_meme.topic,
        "format": db_meme.format,
        "text": db_meme.text,
        "image_url": db_meme.image_url,
        "created_at": db_meme.created_at.isoformat(),
        "score": db_meme.score
    }

@app.get("/api/v1/memes/{meme_id}")
def get_meme(meme_id: str, db: Session = Depends(get_db)):
    """Get a meme by ID.
    
    Args:
        meme_id: Meme ID
        db: Database session
        
    Returns:
        Meme data
    """
    meme = db.query(MemeDB).filter(MemeDB.id == meme_id).first()
    if not meme:
        raise HTTPException(status_code=404, detail="Meme not found")
    
    return {
        "id": meme.id,
        "topic": meme.topic,
        "format": meme.format,
        "text": meme.text,
        "image_url": meme.image_url,
        "created_at": meme.created_at.isoformat(),
        "score": meme.score
    } 