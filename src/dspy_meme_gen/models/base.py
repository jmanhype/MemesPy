"""Base model for SQLAlchemy models."""

from typing import Any, Dict

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BaseModel(Base):
    """Base model class with common functionality."""

    __abstract__ = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
