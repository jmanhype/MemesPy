"""Tests for database exceptions."""
from typing import TYPE_CHECKING

import pytest

from dspy_meme_gen.exceptions.database import (
    CheckConstraintError,
    DatabaseConnectionError,
    DatabaseError,
    DataIntegrityError,
    EntityNotFoundError,
    ForeignKeyError,
    InvalidDataError,
    TransactionError,
    UniqueConstraintError,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_database_error():
    """Test DatabaseError exception."""
    message = "Database error occurred"
    details = {"error_code": 500}
    error = DatabaseError(message, details)

    assert str(error) == message
    assert error.message == message
    assert error.details == details


def test_database_error_no_details():
    """Test DatabaseError with no details."""
    message = "Database error occurred"
    error = DatabaseError(message)

    assert str(error) == message
    assert error.message == message
    assert error.details == {}


def test_entity_not_found_error():
    """Test EntityNotFoundError exception."""
    entity_type = "MemeTemplate"
    entity_id = 123
    error = EntityNotFoundError(entity_type, entity_id)

    assert str(error) == f"{entity_type} with ID {entity_id} not found"
    assert error.details == {"entity_type": entity_type, "entity_id": entity_id}


def test_invalid_data_error():
    """Test InvalidDataError exception."""
    message = "Invalid data provided"
    validation_errors = {"field": "Invalid value"}
    error = InvalidDataError(message, validation_errors)

    assert str(error) == message
    assert error.details == {"validation_errors": validation_errors}


def test_invalid_data_error_no_validation_errors():
    """Test InvalidDataError with no validation errors."""
    message = "Invalid data provided"
    error = InvalidDataError(message)

    assert str(error) == message
    assert error.details == {"validation_errors": {}}


def test_database_connection_error():
    """Test DatabaseConnectionError exception."""
    message = "Failed to connect to database"
    connection_details = {"host": "localhost", "port": 5432}
    error = DatabaseConnectionError(message, connection_details)

    assert str(error) == message
    assert error.details == {"connection_details": connection_details}


def test_database_connection_error_no_details():
    """Test DatabaseConnectionError with no connection details."""
    message = "Failed to connect to database"
    error = DatabaseConnectionError(message)

    assert str(error) == message
    assert error.details == {"connection_details": {}}


def test_transaction_error():
    """Test TransactionError exception."""
    message = "Transaction failed"
    transaction_details = {"transaction_id": "abc123"}
    error = TransactionError(message, transaction_details)

    assert str(error) == message
    assert error.details == {"transaction_details": transaction_details}


def test_transaction_error_no_details():
    """Test TransactionError with no transaction details."""
    message = "Transaction failed"
    error = TransactionError(message)

    assert str(error) == message
    assert error.details == {"transaction_details": {}}


def test_unique_constraint_error():
    """Test UniqueConstraintError exception."""
    message = "Unique constraint violated"
    constraint_details = {"field": "name", "value": "test"}
    error = UniqueConstraintError(message, constraint_details)

    assert str(error) == message
    assert error.details == {"constraint_details": constraint_details}


def test_unique_constraint_error_no_details():
    """Test UniqueConstraintError with no constraint details."""
    message = "Unique constraint violated"
    error = UniqueConstraintError(message)

    assert str(error) == message
    assert error.details == {"constraint_details": {}}


def test_foreign_key_error():
    """Test ForeignKeyError exception."""
    message = "Foreign key constraint violated"
    foreign_key_details = {"table": "memes", "field": "template_id"}
    error = ForeignKeyError(message, foreign_key_details)

    assert str(error) == message
    assert error.details == {"foreign_key_details": foreign_key_details}


def test_foreign_key_error_no_details():
    """Test ForeignKeyError with no foreign key details."""
    message = "Foreign key constraint violated"
    error = ForeignKeyError(message)

    assert str(error) == message
    assert error.details == {"foreign_key_details": {}}


def test_check_constraint_error():
    """Test CheckConstraintError exception."""
    message = "Check constraint violated"
    constraint_details = {"constraint": "rating_range", "value": 6}
    error = CheckConstraintError(message, constraint_details)

    assert str(error) == message
    assert error.details == {"constraint_details": constraint_details}


def test_check_constraint_error_no_details():
    """Test CheckConstraintError with no constraint details."""
    message = "Check constraint violated"
    error = CheckConstraintError(message)

    assert str(error) == message
    assert error.details == {"constraint_details": {}}


def test_data_integrity_error():
    """Test DataIntegrityError exception."""
    message = "Data integrity violation"
    integrity_details = {"table": "memes", "violation": "missing required field"}
    error = DataIntegrityError(message, integrity_details)

    assert str(error) == message
    assert error.details == {"integrity_details": integrity_details}


def test_data_integrity_error_no_details():
    """Test DataIntegrityError with no integrity details."""
    message = "Data integrity violation"
    error = DataIntegrityError(message)

    assert str(error) == message
    assert error.details == {"integrity_details": {}}


def test_exception_inheritance():
    """Test exception inheritance hierarchy."""
    # All exceptions should inherit from DatabaseError
    exceptions = [
        EntityNotFoundError("Test", 1),
        InvalidDataError("Test"),
        DatabaseConnectionError("Test"),
        TransactionError("Test"),
        UniqueConstraintError("Test"),
        ForeignKeyError("Test"),
        CheckConstraintError("Test"),
        DataIntegrityError("Test"),
    ]

    for error in exceptions:
        assert isinstance(error, DatabaseError)
        assert isinstance(error, Exception) 