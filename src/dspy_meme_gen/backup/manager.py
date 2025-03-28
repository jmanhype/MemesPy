"""Backup manager for the DSPy Meme Generation service."""

from typing import Dict, List, Optional, Union
import os
import asyncio
import boto3
from datetime import datetime, timedelta
import subprocess
import shutil
import tempfile
from pathlib import Path
import logging
from botocore.exceptions import BotoCoreError

logger = logging.getLogger(__name__)

class BackupManager:
    """Manages backups for database and media files."""
    
    def __init__(
        self,
        config: Dict[str, Dict[str, Union[str, int]]],
        db_url: str
    ) -> None:
        """Initialize the backup manager.
        
        Args:
            config: Backup configuration
            db_url: Database connection URL
        """
        self.config = config
        self.db_url = db_url
        self.s3_client = boto3.client('s3')
        
    async def create_database_backup(self) -> bool:
        """Create a database backup.
        
        Returns:
            bool: True if backup was successful
        """
        try:
            # Create temporary directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_file = f"db_backup_{timestamp}.sql"
                backup_path = os.path.join(temp_dir, backup_file)
                
                # Extract database connection details
                db_params = self._parse_db_url()
                
                # Create backup using pg_dump
                cmd = [
                    'pg_dump',
                    '-h', db_params['host'],
                    '-p', db_params['port'],
                    '-U', db_params['user'],
                    '-F', 'c',  # Custom format
                    '-b',  # Include large objects
                    '-v',  # Verbose
                    '-f', backup_path,
                    db_params['dbname']
                ]
                
                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                env['PGPASSWORD'] = db_params['password']
                
                # Run pg_dump
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Database backup failed: {stderr.decode()}")
                    return False
                
                # Upload to S3
                bucket = self.config['database']['storage']['bucket']
                s3_path = f"{self.config['database']['storage']['path']}/{backup_file}"
                
                await self._upload_to_s3(backup_path, bucket, s3_path)
                
                # Clean up old backups
                await self._cleanup_old_backups('database')
                
                logger.info(f"Database backup created successfully: {s3_path}")
                return True
                
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return False
            
    async def create_media_backup(self) -> bool:
        """Create a backup of media files.
        
        Returns:
            bool: True if backup was successful
        """
        try:
            # Create temporary directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_file = f"media_backup_{timestamp}.tar.gz"
                backup_path = os.path.join(temp_dir, backup_file)
                
                # Create tar archive of media directory
                media_dir = "media"  # Adjust path as needed
                if os.path.exists(media_dir):
                    process = await asyncio.create_subprocess_exec(
                        'tar', 'czf', backup_path, media_dir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"Media backup failed: {stderr.decode()}")
                        return False
                    
                    # Upload to S3
                    bucket = self.config['media']['storage']['bucket']
                    s3_path = f"{self.config['media']['storage']['path']}/{backup_file}"
                    
                    await self._upload_to_s3(backup_path, bucket, s3_path)
                    
                    # Clean up old backups
                    await self._cleanup_old_backups('media')
                    
                    logger.info(f"Media backup created successfully: {s3_path}")
                    return True
                else:
                    logger.warning(f"Media directory not found: {media_dir}")
                    return False
                    
        except Exception as e:
            logger.error(f"Media backup failed: {str(e)}")
            return False
            
    async def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup.
        
        Args:
            backup_path: S3 path to the backup file
            
        Returns:
            bool: True if restore was successful
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, "db_backup.sql")
                
                # Download from S3
                bucket = self.config['database']['storage']['bucket']
                await self._download_from_s3(bucket, backup_path, local_path)
                
                # Extract database connection details
                db_params = self._parse_db_url()
                
                # Restore using pg_restore
                cmd = [
                    'pg_restore',
                    '-h', db_params['host'],
                    '-p', db_params['port'],
                    '-U', db_params['user'],
                    '-d', db_params['dbname'],
                    '-v',  # Verbose
                    '--clean',  # Clean (drop) database objects before recreating
                    '--if-exists',  # Don't error if objects don't exist
                    local_path
                ]
                
                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                env['PGPASSWORD'] = db_params['password']
                
                # Run pg_restore
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Database restore failed: {stderr.decode()}")
                    return False
                
                logger.info("Database restored successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database restore failed: {str(e)}")
            return False
            
    async def restore_media(self, backup_path: str) -> bool:
        """Restore media files from backup.
        
        Args:
            backup_path: S3 path to the backup file
            
        Returns:
            bool: True if restore was successful
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, "media_backup.tar.gz")
                
                # Download from S3
                bucket = self.config['media']['storage']['bucket']
                await self._download_from_s3(bucket, backup_path, local_path)
                
                # Extract tar archive
                process = await asyncio.create_subprocess_exec(
                    'tar', 'xzf', local_path, '-C', temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Media restore failed: {stderr.decode()}")
                    return False
                
                # Move extracted files to media directory
                media_dir = "media"  # Adjust path as needed
                extracted_dir = os.path.join(temp_dir, "media")
                
                if os.path.exists(media_dir):
                    shutil.rmtree(media_dir)
                shutil.move(extracted_dir, media_dir)
                
                logger.info("Media files restored successfully")
                return True
                
        except Exception as e:
            logger.error(f"Media restore failed: {str(e)}")
            return False
            
    async def _upload_to_s3(self, local_path: str, bucket: str, s3_path: str) -> None:
        """Upload file to S3.
        
        Args:
            local_path: Path to local file
            bucket: S3 bucket name
            s3_path: S3 object key
        """
        try:
            await asyncio.to_thread(
                self.s3_client.upload_file,
                local_path,
                bucket,
                s3_path
            )
        except BotoCoreError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise
            
    async def _download_from_s3(self, bucket: str, s3_path: str, local_path: str) -> None:
        """Download file from S3.
        
        Args:
            bucket: S3 bucket name
            s3_path: S3 object key
            local_path: Path to save file locally
        """
        try:
            await asyncio.to_thread(
                self.s3_client.download_file,
                bucket,
                s3_path,
                local_path
            )
        except BotoCoreError as e:
            logger.error(f"S3 download failed: {str(e)}")
            raise
            
    async def _cleanup_old_backups(self, backup_type: str) -> None:
        """Clean up old backups based on retention policy.
        
        Args:
            backup_type: Type of backup (database or media)
        """
        try:
            retention_days = self.config[backup_type]['retention_days']
            bucket = self.config[backup_type]['storage']['bucket']
            prefix = self.config[backup_type]['storage']['path']
            
            # List objects in bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                            await asyncio.to_thread(
                                self.s3_client.delete_object,
                                Bucket=bucket,
                                Key=obj['Key']
                            )
                            logger.info(f"Deleted old backup: {obj['Key']}")
                            
        except BotoCoreError as e:
            logger.error(f"Cleanup of old backups failed: {str(e)}")
            raise
            
    def _parse_db_url(self) -> Dict[str, str]:
        """Parse database URL into components.
        
        Returns:
            Dict containing database connection parameters
        """
        # Example URL: postgresql://user:pass@host:5432/dbname
        parts = self.db_url.split('://', 1)[1].split('@')
        user_pass = parts[0].split(':')
        host_port_db = parts[1].split('/')
        host_port = host_port_db[0].split(':')
        
        return {
            'user': user_pass[0],
            'password': user_pass[1],
            'host': host_port[0],
            'port': host_port[1] if len(host_port) > 1 else '5432',
            'dbname': host_port_db[1]
        } 