"""Backup-specific exceptions."""

from typing import Optional
from .base import MemeGenerationError


class BackupError(MemeGenerationError):
    """Base class for backup-related errors.

    Attributes:
        message: Error message
        backup_id: Optional backup ID
        operation: Operation that failed
    """

    def __init__(
        self, message: str, backup_id: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message
            backup_id: Optional backup ID
            operation: Operation that failed
        """
        super().__init__(message)
        self.backup_id = backup_id
        self.operation = operation

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            Error message with details
        """
        details = []
        if self.backup_id:
            details.append(f"backup_id={self.backup_id}")
        if self.operation:
            details.append(f"operation={self.operation}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class BackupCreationError(BackupError):
    """Error raised when backup creation fails."""

    def __init__(self, message: str, backup_id: Optional[str] = None) -> None:
        """Initialize the error.

        Args:
            message: Error message
            backup_id: Optional backup ID
        """
        super().__init__(message, backup_id=backup_id, operation="create")


class BackupRestoreError(BackupError):
    """Error raised when backup restoration fails."""

    def __init__(self, message: str, backup_id: str) -> None:
        """Initialize the error.

        Args:
            message: Error message
            backup_id: Backup ID
        """
        super().__init__(message, backup_id=backup_id, operation="restore")


class BackupListError(BackupError):
    """Error raised when listing backups fails."""

    def __init__(self, message: str) -> None:
        """Initialize the error.

        Args:
            message: Error message
        """
        super().__init__(message, operation="list")


class BackupDeleteError(BackupError):
    """Error raised when backup deletion fails."""

    def __init__(self, message: str, backup_id: str) -> None:
        """Initialize the error.

        Args:
            message: Error message
            backup_id: Backup ID
        """
        super().__init__(message, backup_id=backup_id, operation="delete")
