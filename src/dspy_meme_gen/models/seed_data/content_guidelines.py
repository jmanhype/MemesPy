"""Seed data for content guidelines."""

from typing import Dict, List

from ..models import GuidelineCategory, ContentGuideline, SeverityLevel


def get_default_categories() -> List[Dict[str, str]]:
    """
    Get default guideline categories.

    Returns:
        List of category dictionaries
    """
    return [
        {"name": "hate_speech", "description": "Content that promotes hatred or discrimination"},
        {"name": "violence", "description": "Content depicting or promoting violence"},
        {"name": "adult_content", "description": "Adult or sexually explicit content"},
        {"name": "harassment", "description": "Content that harasses or bullies individuals"},
        {"name": "misinformation", "description": "False or misleading information"},
        {
            "name": "copyright",
            "description": "Content that violates copyright or intellectual property",
        },
    ]


def get_default_guidelines() -> List[Dict[str, str]]:
    """
    Get default content guidelines.

    Returns:
        List of guideline dictionaries
    """
    return [
        # Hate Speech Guidelines
        {
            "category_name": "hate_speech",
            "rule": "No content targeting protected characteristics",
            "description": "Content must not discriminate based on race, ethnicity, religion, etc.",
            "severity": SeverityLevel.HIGH,
            "keywords": ["slur", "racist", "discrimination", "hate"],
        },
        {
            "category_name": "hate_speech",
            "rule": "No hate symbols or imagery",
            "description": "Content must not include recognized hate symbols or imagery",
            "severity": SeverityLevel.HIGH,
            "keywords": ["swastika", "hate symbol", "supremacy"],
        },
        # Violence Guidelines
        {
            "category_name": "violence",
            "rule": "No graphic violence",
            "description": "Content must not depict graphic violence or gore",
            "severity": SeverityLevel.HIGH,
            "keywords": ["gore", "blood", "graphic violence", "death"],
        },
        {
            "category_name": "violence",
            "rule": "No promotion of violence",
            "description": "Content must not encourage or promote violent acts",
            "severity": SeverityLevel.HIGH,
            "keywords": ["kill", "attack", "fight", "weapon"],
        },
        # Adult Content Guidelines
        {
            "category_name": "adult_content",
            "rule": "No explicit sexual content",
            "description": "Content must not include explicit sexual material",
            "severity": SeverityLevel.HIGH,
            "keywords": ["nude", "explicit", "sexual", "pornographic"],
        },
        {
            "category_name": "adult_content",
            "rule": "No suggestive content",
            "description": "Content should avoid overly suggestive themes",
            "severity": SeverityLevel.MEDIUM,
            "keywords": ["suggestive", "innuendo", "lewd"],
        },
        # Harassment Guidelines
        {
            "category_name": "harassment",
            "rule": "No personal attacks",
            "description": "Content must not target or harass individuals",
            "severity": SeverityLevel.HIGH,
            "keywords": ["bully", "harass", "mock", "insult"],
        },
        {
            "category_name": "harassment",
            "rule": "No doxxing",
            "description": "Content must not reveal private information",
            "severity": SeverityLevel.HIGH,
            "keywords": ["personal info", "private", "dox", "address"],
        },
        # Misinformation Guidelines
        {
            "category_name": "misinformation",
            "rule": "No false health claims",
            "description": "Content must not make false health or medical claims",
            "severity": SeverityLevel.HIGH,
            "keywords": ["cure", "miracle", "treatment", "vaccine"],
        },
        {
            "category_name": "misinformation",
            "rule": "No conspiracy theories",
            "description": "Content must not promote unfounded conspiracy theories",
            "severity": SeverityLevel.MEDIUM,
            "keywords": ["conspiracy", "hoax", "fake", "theory"],
        },
        # Copyright Guidelines
        {
            "category_name": "copyright",
            "rule": "No unauthorized copyrighted material",
            "description": "Content must respect copyright and intellectual property rights",
            "severity": SeverityLevel.MEDIUM,
            "keywords": ["copyright", "trademark", "stolen", "pirated"],
        },
        {
            "category_name": "copyright",
            "rule": "Fair use compliance",
            "description": "Content must comply with fair use principles when using others' work",
            "severity": SeverityLevel.MEDIUM,
            "keywords": ["fair use", "attribution", "credit", "permission"],
        },
    ]


async def seed_guidelines(session) -> None:
    """
    Seed the database with default content guidelines.

    Args:
        session: SQLAlchemy async session
    """
    # Create categories
    for category_data in get_default_categories():
        category = GuidelineCategory(
            name=category_data["name"], description=category_data["description"]
        )
        session.add(category)

    # Flush to ensure categories are created
    await session.flush()

    # Create guidelines
    for guideline_data in get_default_guidelines():
        # Get category
        category = await session.execute(
            GuidelineCategory.__table__.select().where(
                GuidelineCategory.name == guideline_data["category_name"]
            )
        )
        category = category.scalar_one()

        guideline = ContentGuideline(
            category_id=category.id,
            rule=guideline_data["rule"],
            description=guideline_data["description"],
            severity=guideline_data["severity"],
            keywords=guideline_data["keywords"],
        )
        session.add(guideline)

    await session.commit()
