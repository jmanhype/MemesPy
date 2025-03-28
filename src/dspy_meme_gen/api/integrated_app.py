"""Integrated FastAPI application with database support."""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.dspy_meme_gen.config.settings import get_settings
from src.dspy_meme_gen.models.schemas.memes import MemeGenerationRequest, MemeResponse, MemeListResponse

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
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
    version="0.1.0",
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handler middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions globally."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
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
        "environment": settings.environment,
        "openai_configured": settings.openai_api_key is not None
    }

@app.get("/api/v1/memes/", response_model=MemeListResponse)
def list_memes(db: Session = Depends(get_db)):
    """List all memes.
    
    Args:
        db: Database session
        
    Returns:
        MemeListResponse: List of memes
    """
    memes = db.query(MemeDB).all()
    meme_responses = [
        MemeResponse(
            id=meme.id,
            topic=meme.topic,
            format=meme.format,
            text=meme.text,
            image_url=meme.image_url,
            created_at=meme.created_at.isoformat(),
            score=meme.score
        )
        for meme in memes
    ]
    
    return MemeListResponse(
        items=meme_responses,
        total=len(meme_responses),
        limit=100,
        offset=0
    )

@app.post("/api/v1/memes/", response_model=MemeResponse, status_code=status.HTTP_201_CREATED)
def create_meme(meme_request: MemeGenerationRequest, db: Session = Depends(get_db)):
    """Create a new meme.
    
    Args:
        meme_request: Meme generation request
        db: Database session
        
    Returns:
        MemeResponse: Created meme
    """
    # Generate example meme
    meme_id = str(uuid.uuid4())
    db_meme = MemeDB(
        id=meme_id,
        topic=meme_request.topic,
        format=meme_request.format,
        text=f"Sample meme text about {meme_request.topic}",
        image_url="https://example.com/sample.jpg",
        created_at=datetime.utcnow(),
        score=0.95
    )
    
    # Store in database
    db.add(db_meme)
    db.commit()
    db.refresh(db_meme)
    
    # Return response
    return MemeResponse(
        id=db_meme.id,
        topic=db_meme.topic,
        format=db_meme.format,
        text=db_meme.text,
        image_url=db_meme.image_url,
        created_at=db_meme.created_at.isoformat(),
        score=db_meme.score
    )

@app.get("/api/v1/memes/{meme_id}", response_model=MemeResponse)
def get_meme(meme_id: str, db: Session = Depends(get_db)):
    """Get a meme by ID.
    
    Args:
        meme_id: Meme ID
        db: Database session
        
    Returns:
        MemeResponse: Meme data
    """
    meme = db.query(MemeDB).filter(MemeDB.id == meme_id).first()
    if not meme:
        raise HTTPException(status_code=404, detail="Meme not found")
    
    return MemeResponse(
        id=meme.id,
        topic=meme.topic,
        format=meme.format,
        text=meme.text,
        image_url=meme.image_url,
        created_at=meme.created_at.isoformat(),
        score=meme.score
    ) 