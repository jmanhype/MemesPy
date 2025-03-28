"""Repository layer for database operations."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

from ..models.meme import MemeTemplate, GeneratedMeme
from ..models.content_guidelines import ContentGuideline
from ..models.trending_topics import TrendingTopic

class MemeRepository:
    """Repository for meme-related database operations."""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize the MemeRepository.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def get_template_by_id(self, template_id: int) -> Optional[MemeTemplate]:
        """
        Get a meme template by ID.
        
        Args:
            template_id: ID of the template to retrieve
            
        Returns:
            Optional[MemeTemplate]: The template if found, None otherwise
        """
        result = await self.session.execute(
            select(MemeTemplate).where(MemeTemplate.id == template_id)
        )
        return result.scalar_one_or_none()
    
    async def get_template_by_name(self, name: str) -> Optional[MemeTemplate]:
        """
        Get a meme template by name.
        
        Args:
            name: Name of the template to retrieve
            
        Returns:
            Optional[MemeTemplate]: The template if found, None otherwise
        """
        result = await self.session.execute(
            select(MemeTemplate).where(MemeTemplate.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_all_templates(self) -> List[MemeTemplate]:
        """
        Get all meme templates.
        
        Returns:
            List[MemeTemplate]: List of all templates
        """
        result = await self.session.execute(select(MemeTemplate))
        return list(result.scalars().all())
    
    async def create_template(self, template_data: Dict[str, Any]) -> MemeTemplate:
        """
        Create a new meme template.
        
        Args:
            template_data: Dictionary containing template data
            
        Returns:
            MemeTemplate: The created template
        """
        template = MemeTemplate(**template_data)
        self.session.add(template)
        await self.session.commit()
        return template
    
    async def update_template(self, template_id: int, template_data: Dict[str, Any]) -> Optional[MemeTemplate]:
        """
        Update a meme template.
        
        Args:
            template_id: ID of the template to update
            template_data: Dictionary containing updated template data
            
        Returns:
            Optional[MemeTemplate]: The updated template if found, None otherwise
        """
        await self.session.execute(
            update(MemeTemplate)
            .where(MemeTemplate.id == template_id)
            .values(**template_data)
        )
        await self.session.commit()
        return await self.get_template_by_id(template_id)
    
    async def delete_template(self, template_id: int) -> bool:
        """
        Delete a meme template.
        
        Args:
            template_id: ID of the template to delete
            
        Returns:
            bool: True if template was deleted, False otherwise
        """
        result = await self.session.execute(
            delete(MemeTemplate).where(MemeTemplate.id == template_id)
        )
        await self.session.commit()
        return result.rowcount > 0
    
    async def create_meme(self, meme_data: Dict[str, Any]) -> GeneratedMeme:
        """
        Create a new generated meme.
        
        Args:
            meme_data: Dictionary containing meme data
            
        Returns:
            GeneratedMeme: The created meme
        """
        meme = GeneratedMeme(**meme_data)
        self.session.add(meme)
        await self.session.commit()
        return meme
    
    async def get_meme_by_id(self, meme_id: int) -> Optional[GeneratedMeme]:
        """
        Get a generated meme by ID.
        
        Args:
            meme_id: ID of the meme to retrieve
            
        Returns:
            Optional[GeneratedMeme]: The meme if found, None otherwise
        """
        result = await self.session.execute(
            select(GeneratedMeme)
            .where(GeneratedMeme.id == meme_id)
            .options(selectinload(GeneratedMeme.template))
        )
        return result.scalar_one_or_none()
    
    async def get_recent_memes(self, limit: int = 10) -> List[GeneratedMeme]:
        """
        Get recently generated memes.
        
        Args:
            limit: Maximum number of memes to retrieve
            
        Returns:
            List[GeneratedMeme]: List of recent memes
        """
        result = await self.session.execute(
            select(GeneratedMeme)
            .options(selectinload(GeneratedMeme.template))
            .order_by(GeneratedMeme.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_content_guidelines(self) -> List[ContentGuideline]:
        """
        Get all content guidelines.
        
        Returns:
            List[ContentGuideline]: List of content guidelines
        """
        result = await self.session.execute(select(ContentGuideline))
        return list(result.scalars().all())
    
    async def get_trending_topics(self, limit: int = 10) -> List[TrendingTopic]:
        """
        Get trending topics.
        
        Args:
            limit: Maximum number of topics to retrieve
            
        Returns:
            List[TrendingTopic]: List of trending topics
        """
        result = await self.session.execute(
            select(TrendingTopic)
            .order_by(TrendingTopic.relevance_score.desc())
            .limit(limit)
        )
        return list(result.scalars().all()) 