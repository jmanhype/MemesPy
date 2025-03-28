"""Database-related exceptions."""
from typing import Any, Dict, Optional


class DatabaseError(Exception):
    """Base class for database-related exceptions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            details: Optional dictionary containing additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EntityNotFoundError(DatabaseError):
    """Raised when a requested entity is not found in the database."""

    def __init__(self, entity_type: str, entity_id: Any) -> None:
        """Initialize the exception.
        
        Args:
            entity_type: The type of entity that was not found (e.g., "MemeTemplate").
            entity_id: The ID of the entity that was not found.
        """
        message = f"{entity_type} with ID {entity_id} not found"
        super().__init__(message, {"entity_type": entity_type, "entity_id": entity_id})


class InvalidDataError(DatabaseError):
    """Raised when invalid data is provided for database operations."""

    def __init__(self, message: str, validation_errors: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            validation_errors: Optional dictionary containing validation error details.
        """
        super().__init__(message, {"validation_errors": validation_errors or {}})


class DatabaseConnectionError(DatabaseError):
    """Raised when there is an error connecting to the database."""

    def __init__(self, message: str, connection_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            connection_details: Optional dictionary containing connection error details.
        """
        super().__init__(message, {"connection_details": connection_details or {}})


class TransactionError(DatabaseError):
    """Raised when there is an error during a database transaction."""

    def __init__(self, message: str, transaction_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            transaction_details: Optional dictionary containing transaction error details.
        """
        super().__init__(message, {"transaction_details": transaction_details or {}})


class UniqueConstraintError(DatabaseError):
    """Raised when a unique constraint is violated."""

    def __init__(self, message: str, constraint_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            constraint_details: Optional dictionary containing constraint violation details.
        """
        super().__init__(message, {"constraint_details": constraint_details or {}})


class ForeignKeyError(DatabaseError):
    """Raised when a foreign key constraint is violated."""

    def __init__(self, message: str, foreign_key_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            foreign_key_details: Optional dictionary containing foreign key violation details.
        """
        super().__init__(message, {"foreign_key_details": foreign_key_details or {}})


class CheckConstraintError(DatabaseError):
    """Raised when a check constraint is violated."""

    def __init__(self, message: str, constraint_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            constraint_details: Optional dictionary containing check constraint violation details.
        """
        super().__init__(message, {"constraint_details": constraint_details or {}})


class DataIntegrityError(DatabaseError):
    """Raised when there is a data integrity violation."""

    def __init__(self, message: str, integrity_details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the exception.
        
        Args:
            message: The error message.
            integrity_details: Optional dictionary containing data integrity violation details.
        """
        super().__init__(message, {"integrity_details": integrity_details or {}}) 