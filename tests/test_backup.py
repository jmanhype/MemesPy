"""Tests for backup manager functionality."""

from typing import Dict, Any, cast, TYPE_CHECKING
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime, timedelta
import os
import tempfile
from pathlib import Path

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from dspy_meme_gen.backup.manager import BackupManager
from dspy_meme_gen.exceptions.backup import BackupError

@pytest.fixture
def backup_config() -> Dict[str, Dict[str, Any]]:
    """Create a test backup configuration.
    
    Returns:
        Backup configuration dictionary
    """
    return {
        "database": {
            "storage": {
                "bucket": "test-backups",
                "path": "database"
            },
            "retention_days": 7
        },
        "media": {
            "storage": {
                "bucket": "test-backups",
                "path": "media"
            },
            "retention_days": 30
        }
    }

@pytest.fixture
def db_url() -> str:
    """Create a test database URL.
    
    Returns:
        Database URL string
    """
    return "postgresql://user:pass@localhost:5432/testdb"

@pytest.fixture
def mock_s3_client(mocker: "MockerFixture") -> MagicMock:
    """Create a mock S3 client.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock S3 client
    """
    mock_client = MagicMock()
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {
                    "Key": "database/old_backup.sql",
                    "LastModified": datetime.utcnow() - timedelta(days=10)
                },
                {
                    "Key": "database/recent_backup.sql",
                    "LastModified": datetime.utcnow() - timedelta(days=1)
                }
            ]
        }
    ]
    mock_client.get_paginator.return_value = mock_paginator
    return mock_client

@pytest.fixture
def backup_manager(backup_config: Dict[str, Dict[str, Any]], db_url: str, mock_s3_client: MagicMock) -> BackupManager:
    """Create a BackupManager instance with mock dependencies.
    
    Args:
        backup_config: Backup configuration
        db_url: Database URL
        mock_s3_client: Mock S3 client
        
    Returns:
        BackupManager instance
    """
    with patch('boto3.client', return_value=mock_s3_client):
        return BackupManager(backup_config, db_url)

@pytest.mark.asyncio
async def test_create_database_backup_success(
    backup_manager: BackupManager,
    mocker: "MockerFixture",
    tmp_path: Path
) -> None:
    """Test successful database backup creation.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
        tmp_path: pytest temporary path fixture
    """
    # Mock subprocess execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Execute backup
    success = await backup_manager.create_database_backup()
    
    assert success is True
    # Verify pg_dump was called with correct arguments
    asyncio.create_subprocess_exec.assert_called_once()  # type: ignore
    call_args = asyncio.create_subprocess_exec.call_args[0]  # type: ignore
    assert call_args[0] == 'pg_dump'
    assert '-h' in call_args and 'localhost' in call_args
    assert '-U' in call_args and 'user' in call_args
    
    # Verify S3 upload
    backup_manager.s3_client.upload_file.assert_called_once()  # type: ignore
    assert backup_manager.s3_client.upload_file.call_args[0][1] == 'test-backups'  # type: ignore

@pytest.mark.asyncio
async def test_create_database_backup_pg_dump_failure(
    backup_manager: BackupManager,
    mocker: "MockerFixture"
) -> None:
    """Test database backup creation when pg_dump fails.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
    """
    # Mock subprocess execution failure
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = (b"", b"Backup failed")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Execute backup
    success = await backup_manager.create_database_backup()
    
    assert success is False
    # Verify S3 upload was not called
    backup_manager.s3_client.upload_file.assert_not_called()  # type: ignore

@pytest.mark.asyncio
async def test_create_media_backup_success(
    backup_manager: BackupManager,
    mocker: "MockerFixture",
    tmp_path: Path
) -> None:
    """Test successful media backup creation.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
        tmp_path: pytest temporary path fixture
    """
    # Create test media directory and file
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    test_file = media_dir / "test.jpg"
    test_file.write_bytes(b"test content")
    
    # Mock os.path.exists
    mocker.patch('os.path.exists', return_value=True)
    
    # Mock subprocess execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Execute backup
    success = await backup_manager.create_media_backup()
    
    assert success is True
    # Verify tar was called
    asyncio.create_subprocess_exec.assert_called_once()  # type: ignore
    call_args = asyncio.create_subprocess_exec.call_args[0]  # type: ignore
    assert call_args[0] == 'tar'
    
    # Verify S3 upload
    backup_manager.s3_client.upload_file.assert_called_once()  # type: ignore
    assert backup_manager.s3_client.upload_file.call_args[0][1] == 'test-backups'  # type: ignore

@pytest.mark.asyncio
async def test_create_media_backup_no_media_dir(
    backup_manager: BackupManager,
    mocker: "MockerFixture"
) -> None:
    """Test media backup creation when media directory doesn't exist.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
    """
    # Mock os.path.exists to return False
    mocker.patch('os.path.exists', return_value=False)
    
    # Execute backup
    success = await backup_manager.create_media_backup()
    
    assert success is False
    # Verify no S3 upload was attempted
    backup_manager.s3_client.upload_file.assert_not_called()  # type: ignore

@pytest.mark.asyncio
async def test_restore_database_success(
    backup_manager: BackupManager,
    mocker: "MockerFixture"
) -> None:
    """Test successful database restore.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
    """
    # Mock subprocess execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Execute restore
    success = await backup_manager.restore_database("database/backup.sql")
    
    assert success is True
    # Verify pg_restore was called
    asyncio.create_subprocess_exec.assert_called_once()  # type: ignore
    call_args = asyncio.create_subprocess_exec.call_args[0]  # type: ignore
    assert call_args[0] == 'pg_restore'
    assert '--clean' in call_args
    assert '--if-exists' in call_args

@pytest.mark.asyncio
async def test_restore_media_success(
    backup_manager: BackupManager,
    mocker: "MockerFixture",
    tmp_path: Path
) -> None:
    """Test successful media restore.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
        tmp_path: pytest temporary path fixture
    """
    # Mock subprocess execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Mock shutil operations
    mocker.patch('shutil.rmtree')
    mocker.patch('shutil.move')
    
    # Execute restore
    success = await backup_manager.restore_media("media/backup.tar.gz")
    
    assert success is True
    # Verify tar extraction was called
    asyncio.create_subprocess_exec.assert_called_once()  # type: ignore
    call_args = asyncio.create_subprocess_exec.call_args[0]  # type: ignore
    assert call_args[0] == 'tar'
    assert 'xzf' in call_args

@pytest.mark.asyncio
async def test_cleanup_old_backups(
    backup_manager: BackupManager,
    mock_s3_client: MagicMock
) -> None:
    """Test cleanup of old backups.
    
    Args:
        backup_manager: BackupManager instance
        mock_s3_client: Mock S3 client
    """
    # Execute cleanup
    await backup_manager.cleanup_old_backups('database')
    
    # Verify old backup was deleted
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket='test-backups',
        Key='database/old_backup.sql'
    )

@pytest.mark.asyncio
async def test_s3_upload_error(
    backup_manager: BackupManager,
    mock_s3_client: MagicMock
) -> None:
    """Test handling of S3 upload errors.
    
    Args:
        backup_manager: BackupManager instance
        mock_s3_client: Mock S3 client
    """
    # Mock S3 upload failure
    mock_s3_client.upload_file.side_effect = BotoCoreError()
    
    with pytest.raises(BotoCoreError):
        await backup_manager._upload_to_s3("test.file", "test-bucket", "test/path")

@pytest.mark.asyncio
async def test_s3_download_error(
    backup_manager: BackupManager,
    mock_s3_client: MagicMock
) -> None:
    """Test handling of S3 download errors.
    
    Args:
        backup_manager: BackupManager instance
        mock_s3_client: Mock S3 client
    """
    # Mock S3 download failure
    mock_s3_client.download_file.side_effect = BotoCoreError()
    
    with pytest.raises(BotoCoreError):
        await backup_manager._download_from_s3("test-bucket", "test/path", "local.file")

@pytest.fixture
def mock_backup_manager(
    mocker: "MockerFixture",
    mock_s3_client: MagicMock
) -> MagicMock:
    """Create a mock backup manager.
    
    Args:
        mocker: pytest-mock fixture
        mock_s3_client: Mock S3 client
        
    Returns:
        Mock backup manager
    """
    mock_manager = mocker.MagicMock()
    mock_manager.s3_client = mock_s3_client
    return mock_manager

@pytest.mark.asyncio
async def test_backup_creation(
    backup_manager: BackupManager,
    mocker: "MockerFixture",
    tmp_path: Path
) -> None:
    """Test successful database backup creation.
    
    Args:
        backup_manager: BackupManager instance
        mocker: pytest-mock fixture
        tmp_path: pytest temporary path fixture
    """
    # Mock subprocess execution
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mocker.patch('asyncio.create_subprocess_exec', return_value=mock_process)
    
    # Execute backup
    success = await backup_manager.create_database_backup()
    
    assert success is True
    # Verify pg_dump was called with correct arguments
    asyncio.create_subprocess_exec.assert_called_once()  # type: ignore
    call_args = asyncio.create_subprocess_exec.call_args[0]  # type: ignore
    assert call_args[0] == 'pg_dump'
    assert '-h' in call_args and 'localhost' in call_args
    assert '-U' in call_args and 'user' in call_args
    
    # Verify S3 upload
    backup_manager.s3_client.upload_file.assert_called_once()  # type: ignore
    assert backup_manager.s3_client.upload_file.call_args[0][1] == 'test-backups'  # type: ignore

async def test_backup_restoration(
    mock_backup_manager: MagicMock,
    mocker: "MockerFixture"
) -> None:
    """Test backup restoration."""
    # Setup test data
    backup_id = "test_backup"
    restored_data = {"key": "value"}
    
    # Configure mock
    mock_backup_manager.restore_backup.return_value = restored_data
    
    # Execute backup restoration
    result = await mock_backup_manager.restore_backup(backup_id)
    
    # Verify result
    assert result == restored_data
    mock_backup_manager.restore_backup.assert_called_once_with(backup_id)

async def test_backup_listing(
    mock_backup_manager: MagicMock,
    mocker: "MockerFixture"
) -> None:
    """Test backup listing."""
    # Setup test data
    backup_list = [
        {"id": "backup1", "timestamp": "2024-03-27T10:00:00Z"},
        {"id": "backup2", "timestamp": "2024-03-27T11:00:00Z"}
    ]
    
    # Configure mock
    mock_backup_manager.list_backups.return_value = backup_list
    
    # Execute backup listing
    result = await mock_backup_manager.list_backups()
    
    # Verify result
    assert result == backup_list
    mock_backup_manager.list_backups.assert_called_once()

async def test_backup_error_handling(
    mock_backup_manager: MagicMock,
    mocker: "MockerFixture"
) -> None:
    """Test backup error handling."""
    # Configure mock to raise an error
    mock_backup_manager.create_backup.side_effect = BackupError("Test error")
    
    # Execute backup creation and verify error handling
    with pytest.raises(BackupError, match="Test error"):
        await mock_backup_manager.create_backup({}) 