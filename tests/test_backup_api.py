"""Tests for backup API endpoints."""

from typing import TYPE_CHECKING, Dict, Any
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
from dspy_meme_gen.api.backup import router, get_backup_manager
from dspy_meme_gen.backup.manager import BackupManager

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture
def app(mocker: "MockerFixture") -> FastAPI:
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router)
    
    # Mock backup manager
    mock_manager = mocker.MagicMock()
    mock_manager.backup = mocker.AsyncMock()
    mock_manager.restore = mocker.AsyncMock()
    mock_manager.list_backups = mocker.AsyncMock()
    
    # Override dependency
    app.dependency_overrides[get_backup_manager] = lambda: mock_manager
    
    return app

@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    """Create test client.
    
    Args:
        app: FastAPI application
        
    Returns:
        TestClient instance
    """
    return TestClient(app)

@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Create a test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_backup_manager() -> AsyncMock:
    """Create mock backup manager.
    
    Returns:
        AsyncMock instance
    """
    manager = AsyncMock(spec=BackupManager)
    manager.create_database_backup = AsyncMock(return_value=True)
    manager.create_media_backup = AsyncMock(return_value=True)
    manager.restore_database = AsyncMock(return_value=True)
    manager.restore_media = AsyncMock(return_value=True)
    manager.cleanup_old_backups = AsyncMock(return_value=True)
    return manager

@pytest.mark.asyncio
async def test_create_database_backup(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test database backup creation.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.post("/backup/database")
        assert response.status_code == 200
        assert response.json() == {"status": "backup_started", "type": "database"}
        mock_backup_manager.create_database_backup.assert_called_once()

@pytest.mark.asyncio
async def test_create_media_backup(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test media backup creation.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.post("/backup/media")
        assert response.status_code == 200
        assert response.json() == {"status": "backup_started", "type": "media"}
        mock_backup_manager.create_media_backup.assert_called_once()

@pytest.mark.asyncio
async def test_restore_database(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test database restore.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    backup_path = "backups/database/backup.sql"
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.post(f"/backup/database/restore/{backup_path}")
        assert response.status_code == 200
        assert response.json() == {
            "status": "restore_started",
            "type": "database",
            "backup_path": backup_path
        }
        mock_backup_manager.restore_database.assert_called_once_with(backup_path)

@pytest.mark.asyncio
async def test_restore_media(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test media restore.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    backup_path = "backups/media/backup.tar.gz"
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.post(f"/backup/media/restore/{backup_path}")
        assert response.status_code == 200
        assert response.json() == {
            "status": "restore_started",
            "type": "media",
            "backup_path": backup_path
        }
        mock_backup_manager.restore_media.assert_called_once_with(backup_path)

@pytest.mark.asyncio
async def test_cleanup_old_backups(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test backup cleanup.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    backup_type = "database"
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.delete(f"/backup/cleanup/{backup_type}")
        assert response.status_code == 200
        assert response.json() == {"status": "cleanup_started", "type": backup_type}
        mock_backup_manager.cleanup_old_backups.assert_called_once_with(backup_type)

@pytest.mark.asyncio
async def test_cleanup_invalid_backup_type(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test cleanup with invalid backup type.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    backup_type = "invalid"
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.delete(f"/backup/cleanup/{backup_type}")
        assert response.status_code == 400
        assert "Invalid backup type" in response.json()["detail"]
        mock_backup_manager.cleanup_old_backups.assert_not_called()

@pytest.mark.asyncio
async def test_backup_failure_handling(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test handling of backup failures.
    
    Args:
        client: AsyncClient instance
        app: FastAPI application
        mocker: MockerFixture instance
    """
    mock_backup_manager.create_database_backup.return_value = False
    with patch("dspy_meme_gen.api.backup.get_backup_manager", return_value=mock_backup_manager):
        response = await client.post("/backup/database")
        assert response.status_code == 200  # Still returns 200 as task is started
        assert response.json() == {"status": "backup_started", "type": "database"}
        mock_backup_manager.create_database_backup.assert_called_once()

async def test_create_backup(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test creating a backup."""
    # Setup test data
    test_data = {"key": "value"}
    expected_backup_id = "backup_123"
    
    # Configure mock
    mock_manager = app.dependency_overrides[get_backup_manager]()
    mock_manager.backup.return_value = expected_backup_id
    
    # Execute request
    response = await client.post("/backup", json=test_data)
    
    # Verify
    assert response.status_code == 200
    assert response.json() == {"backup_id": expected_backup_id}
    mock_manager.backup.assert_called_once_with(test_data)

async def test_restore_backup(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test restoring from a backup."""
    # Setup test data
    backup_id = "backup_123"
    expected_data = {"key": "value"}
    
    # Configure mock
    mock_manager = app.dependency_overrides[get_backup_manager]()
    mock_manager.restore.return_value = expected_data
    
    # Execute request
    response = await client.get(f"/backup/{backup_id}")
    
    # Verify
    assert response.status_code == 200
    assert response.json() == expected_data
    mock_manager.restore.assert_called_once_with(backup_id)

async def test_list_backups(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test listing backups."""
    # Setup test data
    expected_backups = [
        {"id": "backup_1", "timestamp": "2024-03-27T12:00:00Z"},
        {"id": "backup_2", "timestamp": "2024-03-27T13:00:00Z"}
    ]
    
    # Configure mock
    mock_manager = app.dependency_overrides[get_backup_manager]()
    mock_manager.list_backups.return_value = expected_backups
    
    # Execute request
    response = await client.get("/backup")
    
    # Verify
    assert response.status_code == 200
    assert response.json() == expected_backups
    mock_manager.list_backups.assert_called_once()

async def test_backup_error_handling(
    client: AsyncClient,
    app: FastAPI,
    mocker: "MockerFixture"
) -> None:
    """Test error handling in backup endpoints."""
    # Configure mock to raise an error
    mock_manager = app.dependency_overrides[get_backup_manager]()
    mock_manager.backup.side_effect = Exception("Backup failed")
    
    # Execute request and verify
    response = await client.post("/backup", json={})
    
    assert response.status_code == 500
    assert "error" in response.json()
    assert "Backup failed" in response.json()["error"] 