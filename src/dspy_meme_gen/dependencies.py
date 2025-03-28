"""Dependencies for FastAPI application."""

from typing import Dict, Any
from functools import lru_cache

from .config.settings import Settings, get_settings
from .backup.manager import BackupManager

@lru_cache()
def get_backup_config() -> Dict[str, Any]:
    """Get backup configuration.
    
    Returns:
        Dictionary containing backup configuration
    """
    settings = get_settings()
    return {
        "bucket_name": settings.s3_bucket_name,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
        "aws_region": settings.aws_region
    }

@lru_cache()
def get_db_url() -> str:
    """Get database URL.
    
    Returns:
        Database URL string
    """
    settings = get_settings()
    return settings.database_url

@lru_cache()
def get_backup_manager() -> BackupManager:
    """Get backup manager instance.
    
    Returns:
        BackupManager instance
    """
    config = get_backup_config()
    return BackupManager(**config) 