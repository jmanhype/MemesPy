"""FastAPI endpoints for meme generation and management."""
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.repository import MemeRepository
from ..models.meme import MemeTemplate, GeneratedMeme, TrendingTopic
from ..utils.config import AppConfig, get_config
from ..utils.image import add_text_overlay, convert_format, resize_image
from ..utils.text import format_meme_text, validate_meme_text

# Request/Response Models
class MemeTemplateCreate(BaseModel):
    """Request model for creating a meme template."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    format_type: str = Field(..., description="Format type (e.g., 'image', 'video')")
    example_url: Optional[str] = Field(None, description="Example URL")
    structure: dict = Field(..., description="Template structure")

class MemeTemplateResponse(BaseModel):
    """Response model for meme template."""
    id: int
    name: str
    description: str
    format_type: str
    example_url: Optional[str]
    structure: dict
    popularity_score: float

class GeneratedMemeCreate(BaseModel):
    """Request model for generating a meme."""
    template_id: int = Field(..., description="Template ID")
    topic: str = Field(..., description="Meme topic")
    caption: str = Field(..., description="Meme caption")
    image_prompt: Optional[str] = Field(None, description="Image generation prompt")

class GeneratedMemeResponse(BaseModel):
    """Response model for generated meme."""
    id: int
    template_id: int
    topic: str
    caption: str
    image_url: str
    score: Optional[float]
    created_at: str

class TrendingTopicResponse(BaseModel):
    """Response model for trending topic."""
    id: int
    topic: str
    source: str
    relevance_score: float
    timestamp: str
    metadata: Optional[dict]

# Router setup
router = APIRouter(prefix="/api/v1")

# Dependencies
async def get_repository(session: AsyncSession) -> MemeRepository:
    """Get repository instance."""
    return MemeRepository(session)

async def get_app_config() -> AppConfig:
    """Get application configuration."""
    return get_config()

# Endpoints
@router.post("/templates", response_model=MemeTemplateResponse)
async def create_template(
    template: MemeTemplateCreate,
    repo: MemeRepository = Depends(get_repository),
    config: AppConfig = Depends(get_app_config)
) -> MemeTemplate:
    """Create a new meme template."""
    try:
        return await repo.create_meme_template(
            name=template.name,
            description=template.description,
            format_type=template.format_type,
            example_url=template.example_url,
            structure=template.structure
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates/{template_id}", response_model=MemeTemplateResponse)
async def get_template(
    template_id: int,
    repo: MemeRepository = Depends(get_repository)
) -> MemeTemplate:
    """Get a meme template by ID."""
    template = await repo.get_meme_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template

@router.get("/templates", response_model=List[MemeTemplateResponse])
async def list_templates(
    format_type: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo: MemeRepository = Depends(get_repository)
) -> List[MemeTemplate]:
    """List meme templates."""
    return await repo.list_meme_templates(
        format_type=format_type,
        limit=limit,
        offset=offset
    )

@router.post("/memes", response_model=GeneratedMemeResponse)
async def generate_meme(
    meme: GeneratedMemeCreate,
    repo: MemeRepository = Depends(get_repository),
    config: AppConfig = Depends(get_app_config)
) -> GeneratedMeme:
    """Generate a new meme."""
    # Validate text
    is_valid, error = validate_meme_text(meme.caption)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Format text
    try:
        formatted_caption = format_meme_text(meme.caption)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get template
    template = await repo.get_meme_template(meme.template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        return await repo.create_generated_meme(
            template_id=meme.template_id,
            topic=meme.topic,
            caption=formatted_caption,
            image_prompt=meme.image_prompt
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/memes/{meme_id}", response_model=GeneratedMemeResponse)
async def get_meme(
    meme_id: int,
    repo: MemeRepository = Depends(get_repository)
) -> GeneratedMeme:
    """Get a generated meme by ID."""
    meme = await repo.get_generated_meme(meme_id)
    if not meme:
        raise HTTPException(status_code=404, detail="Meme not found")
    return meme

@router.get("/memes", response_model=List[GeneratedMemeResponse])
async def list_memes(
    template_id: Optional[int] = None,
    topic: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo: MemeRepository = Depends(get_repository)
) -> List[GeneratedMeme]:
    """List generated memes."""
    return await repo.list_generated_memes(
        template_id=template_id,
        topic=topic,
        limit=limit,
        offset=offset
    )

@router.get("/trending", response_model=List[TrendingTopicResponse])
async def list_trending(
    source: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo: MemeRepository = Depends(get_repository)
) -> List[TrendingTopic]:
    """List trending topics."""
    return await repo.list_trending_topics(
        source=source,
        limit=limit,
        offset=offset
    )

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    config: AppConfig = Depends(get_app_config)
) -> dict:
    """Upload an image file."""
    # Validate file size
    if file.size > config.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {config.max_file_size} bytes"
        )
    
    # Validate format
    format = file.filename.split(".")[-1].upper()
    if format not in config.allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Allowed formats: {', '.join(config.allowed_formats)}"
        )
    
    try:
        # Process image
        image = await file.read()
        processed = resize_image(image, (800, 600))
        
        # TODO: Upload to storage (e.g., Cloudinary)
        # For now, return dummy URL
        return {"url": "https://example.com/image.jpg"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feedback/{meme_id}")
async def add_feedback(
    meme_id: int,
    score: float = Query(..., ge=0, le=1),
    repo: MemeRepository = Depends(get_repository)
) -> dict:
    """Add user feedback for a meme."""
    try:
        await repo.add_user_feedback(meme_id, score)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 