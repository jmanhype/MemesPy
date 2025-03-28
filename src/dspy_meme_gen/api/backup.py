"""Backup API endpoints."""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from ..backup.manager import BackupManager
from ..dependencies import get_backup_config, get_db_url
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backup", tags=["backup"])

async def get_backup_manager(
    config: Dict[str, Dict[str, Any]] = Depends(get_backup_config),
    db_url: str = Depends(get_db_url)
) -> BackupManager:
    """Get BackupManager instance.
    
    Args:
        config: Backup configuration
        db_url: Database URL
        
    Returns:
        BackupManager instance
    """
    return BackupManager(config, db_url)

@router.post(
    "/database",
    response_model=Dict[str, Any],
    summary="Create database backup",
    description="Create a backup of the database and store it in S3."
)
async def create_database_backup(
    background_tasks: BackgroundTasks,
    backup_manager: BackupManager = Depends(get_backup_manager)
) -> Dict[str, Any]:
    """Create database backup.
    
    Args:
        background_tasks: FastAPI background tasks
        backup_manager: BackupManager instance
        
    Returns:
        Status message
    """
    async def _create_backup() -> None:
        try:
            success = await backup_manager.create_database_backup()
            if not success:
                logger.error("Database backup failed")
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
    
    background_tasks.add_task(_create_backup)
    return {"status": "backup_started", "type": "database"}

@router.post(
    "/media",
    response_model=Dict[str, Any],
    summary="Create media backup",
    description="Create a backup of media files and store it in S3."
)
async def create_media_backup(
    background_tasks: BackgroundTasks,
    backup_manager: BackupManager = Depends(get_backup_manager)
) -> Dict[str, Any]:
    """Create media backup.
    
    Args:
        background_tasks: FastAPI background tasks
        backup_manager: BackupManager instance
        
    Returns:
        Status message
    """
    async def _create_backup() -> None:
        try:
            success = await backup_manager.create_media_backup()
            if not success:
                logger.error("Media backup failed")
        except Exception as e:
            logger.error(f"Media backup failed: {str(e)}")
    
    background_tasks.add_task(_create_backup)
    return {"status": "backup_started", "type": "media"}

@router.post(
    "/database/restore/{backup_path:path}",
    response_model=Dict[str, Any],
    summary="Restore database from backup",
    description="Restore the database from a specified backup in S3."
)
async def restore_database(
    backup_path: str,
    background_tasks: BackgroundTasks,
    backup_manager: BackupManager = Depends(get_backup_manager)
) -> Dict[str, Any]:
    """Restore database from backup.
    
    Args:
        backup_path: Path to backup file in S3
        background_tasks: FastAPI background tasks
        backup_manager: BackupManager instance
        
    Returns:
        Status message
    """
    async def _restore_backup() -> None:
        try:
            success = await backup_manager.restore_database(backup_path)
            if not success:
                logger.error("Database restore failed")
        except Exception as e:
            logger.error(f"Database restore failed: {str(e)}")
    
    background_tasks.add_task(_restore_backup)
    return {"status": "restore_started", "type": "database", "backup_path": backup_path}

@router.post(
    "/media/restore/{backup_path:path}",
    response_model=Dict[str, Any],
    summary="Restore media from backup",
    description="Restore media files from a specified backup in S3."
)
async def restore_media(
    backup_path: str,
    background_tasks: BackgroundTasks,
    backup_manager: BackupManager = Depends(get_backup_manager)
) -> Dict[str, Any]:
    """Restore media from backup.
    
    Args:
        backup_path: Path to backup file in S3
        background_tasks: FastAPI background tasks
        backup_manager: BackupManager instance
        
    Returns:
        Status message
    """
    async def _restore_backup() -> None:
        try:
            success = await backup_manager.restore_media(backup_path)
            if not success:
                logger.error("Media restore failed")
        except Exception as e:
            logger.error(f"Media restore failed: {str(e)}")
    
    background_tasks.add_task(_restore_backup)
    return {"status": "restore_started", "type": "media", "backup_path": backup_path}

@router.delete(
    "/cleanup/{backup_type}",
    response_model=Dict[str, Any],
    summary="Clean up old backups",
    description="Delete old backups based on retention policy."
)
async def cleanup_old_backups(
    backup_type: str,
    background_tasks: BackgroundTasks,
    backup_manager: BackupManager = Depends(get_backup_manager)
) -> Dict[str, Any]:
    """Clean up old backups.
    
    Args:
        backup_type: Type of backup (database or media)
        background_tasks: FastAPI background tasks
        backup_manager: BackupManager instance
        
    Returns:
        Status message
        
    Raises:
        HTTPException: If backup type is invalid
    """
    if backup_type not in ["database", "media"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backup type: {backup_type}"
        )
    
    async def _cleanup_backups() -> None:
        try:
            await backup_manager.cleanup_old_backups(backup_type)
        except Exception as e:
            logger.error(f"Backup cleanup failed: {str(e)}")
    
    background_tasks.add_task(_cleanup_backups)
    return {"status": "cleanup_started", "type": backup_type} 