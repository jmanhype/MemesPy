"""Settings module for the DSPy Meme Generator."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings.
    
    Attributes:
        database_url: Database connection URL
        redis_url: Redis connection URL
        openai_api_key: OpenAI API key for image generation
        cloudinary_cloud_name: Cloudinary cloud name
        cloudinary_api_key: Cloudinary API key
        cloudinary_api_secret: Cloudinary API secret
        s3_bucket_name: S3 bucket name for backups
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_region: AWS region
        log_level: Logging level
        environment: Application environment
    """
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    # Database settings
    database_url: str
    redis_url: Optional[str] = None
    
    # API keys
    openai_api_key: str
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str
    
    # AWS settings
    s3_bucket_name: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # Application settings
    log_level: str = "INFO"
    environment: str = "development"

def get_settings() -> Settings:
    """Get application settings.
    
    Returns:
        Settings instance
    """
    return Settings() 