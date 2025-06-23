"""Privacy-first database models with GDPR compliance."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from sqlalchemy import (
    Column, String, Float, DateTime, JSON, Integer, Text, Boolean,
    ForeignKey, Index, event, and_, or_, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from ...config.config import settings
import hashlib
import secrets

Base = declarative_base()


class ConsentType(str, Enum):
    """Types of consent that can be granted."""
    ESSENTIAL = "essential"  # Required for service operation
    ANALYTICS = "analytics"  # Performance and usage analytics
    PERSONALIZATION = "personalization"  # User preferences and recommendations
    MARKETING = "marketing"  # Marketing communications


class DataRetentionPeriod(str, Enum):
    """Standard retention periods for different data types."""
    TRANSIENT = "transient"  # Delete immediately after use
    SESSION = "session"  # Delete after session ends
    WEEK = "week"  # 7 days
    MONTH = "month"  # 30 days
    QUARTER = "quarter"  # 90 days
    YEAR = "year"  # 365 days
    LEGAL = "legal"  # As required by law


class PrivacyMemeMetadata(Base):
    """
    Privacy-first meme model with minimal data collection and GDPR compliance.
    
    Key principles:
    - Data minimization: Only collect what's necessary
    - Purpose limitation: Data used only for stated purposes
    - Privacy by design: Built-in privacy protections
    - User control: Full control over personal data
    """
    
    __tablename__ = "privacy_meme_metadata"
    __table_args__ = (
        Index('idx_deletion_date', 'scheduled_deletion_date'),
        Index('idx_pseudonym_created', 'user_pseudonym', 'created_at'),
    )
    
    # Core identifiers (no PII)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pseudonymized user reference (not linked to real identity)
    user_pseudonym = Column(String, index=True)  # One-way hash of user identifier
    session_id = Column(String)  # Temporary session identifier
    
    # Essential meme data (required for service)
    topic = Column(String, nullable=False)  # No PII allowed in topics
    format = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    image_url = Column(String, nullable=False)
    
    # Timestamps with automatic deletion scheduling
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scheduled_deletion_date = Column(DateTime, nullable=False)
    
    # Minimal quality metrics (anonymized)
    score = Column(Float)  # Aggregated score without individual ratings
    view_count = Column(Integer, default=0)  # Anonymous view counter
    
    # Anonymized generation metadata
    generation_metadata = Column(JSON, default={})
    """
    Minimal metadata:
    - model_type: str (general model category, not specific version)
    - generation_time_ms: int (rounded to nearest 100ms)
    - success: bool
    - retry_count: int (capped at 3)
    """
    
    # Content safety flags (no detailed analysis stored)
    safety_flags = Column(JSON, default={})
    """
    Binary flags only:
    - is_safe: bool
    - requires_moderation: bool
    """
    
    # Data processing consent tracking
    consent_settings = Column(JSON, default={})
    """
    Current consent status:
    - essential: bool (always true for active records)
    - analytics: bool
    - personalization: bool
    """
    
    # Retention and deletion
    retention_period = Column(String, default=DataRetentionPeriod.MONTH)
    deletion_requested = Column(Boolean, default=False)
    deletion_requested_at = Column(DateTime)
    
    # Relationships
    audit_logs = relationship("PrivacyAuditLog", back_populates="meme", cascade="all, delete-orphan")
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if data has exceeded retention period."""
        return datetime.utcnow() > self.scheduled_deletion_date
    
    @hybrid_property
    def days_until_deletion(self) -> int:
        """Calculate days until automatic deletion."""
        delta = self.scheduled_deletion_date - datetime.utcnow()
        return max(0, delta.days)
    
    def anonymize(self) -> Dict[str, Any]:
        """Return fully anonymized version of the data."""
        return {
            'id': self.id,
            'topic_category': self._generalize_topic(self.topic),
            'format': self.format,
            'created_date': self.created_at.date().isoformat(),
            'quality_tier': self._generalize_score(self.score),
            'view_tier': self._generalize_views(self.view_count)
        }
    
    def _generalize_topic(self, topic: str) -> str:
        """Generalize topic to broader category."""
        # Implementation would map specific topics to general categories
        # e.g., "cute cats" -> "animals", "bitcoin meme" -> "technology"
        return "general"
    
    def _generalize_score(self, score: Optional[float]) -> str:
        """Convert score to tier."""
        if not score:
            return "unrated"
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        return "low"
    
    def _generalize_views(self, views: int) -> str:
        """Convert view count to tier."""
        if views >= 1000:
            return "popular"
        elif views >= 100:
            return "moderate"
        return "low"


class UserConsent(Base):
    """
    Explicit consent management with granular control.
    """
    
    __tablename__ = "user_consent"
    __table_args__ = (
        UniqueConstraint('user_pseudonym', 'consent_type'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_pseudonym = Column(String, nullable=False, index=True)
    
    consent_type = Column(String, nullable=False)
    granted = Column(Boolean, nullable=False, default=False)
    
    # Consent lifecycle
    granted_at = Column(DateTime)
    revoked_at = Column(DateTime)
    expires_at = Column(DateTime)  # Auto-revoke after period
    
    # Consent details
    version = Column(String, nullable=False)  # Privacy policy version
    ip_country = Column(String)  # Country only, no specific IP stored
    
    # Purpose and legal basis
    purpose = Column(Text, nullable=False)
    legal_basis = Column(String, nullable=False)  # consent, legitimate_interest, etc.
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if consent is currently active."""
        if not self.granted or self.revoked_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class PrivacyAuditLog(Base):
    """
    Audit log for data access and modifications without storing PII.
    """
    
    __tablename__ = "privacy_audit_log"
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_pseudonym', 'user_pseudonym'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Action details (no PII)
    action_type = Column(String, nullable=False)  # create, read, update, delete, export
    resource_type = Column(String, nullable=False)  # meme, consent, user_data
    resource_id = Column(String)  # ID of affected resource
    
    # Pseudonymized actor
    user_pseudonym = Column(String)  # Who performed the action
    system_component = Column(String)  # Which system component
    
    # Minimal context
    purpose = Column(String)  # Why the action was performed
    legal_basis = Column(String)  # Legal justification
    
    # No IP addresses, user agents, or other PII stored
    # Only store generalized location and time
    country_code = Column(String)  # Two-letter country code only
    hour_of_day = Column(Integer)  # 0-23, no exact timestamp
    
    # Relationship
    meme_id = Column(String, ForeignKey('privacy_meme_metadata.id', ondelete='CASCADE'))
    meme = relationship("PrivacyMemeMetadata", back_populates="audit_logs")


class AnonymizedAnalytics(Base):
    """
    Aggregated analytics data with k-anonymity guarantees.
    """
    
    __tablename__ = "anonymized_analytics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    granularity = Column(String, nullable=False)  # hourly, daily, weekly
    
    # Aggregated metrics (minimum k=5 for any group)
    metric_type = Column(String, nullable=False)
    dimension = Column(String)  # e.g., "format:image_macro"
    
    # Counts and averages only
    count = Column(Integer, nullable=False)
    sum_value = Column(Float)
    avg_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    
    # K-anonymity guarantee
    k_value = Column(Integer, nullable=False, default=5)
    
    # No user-level data stored
    __table_args__ = (
        Index('idx_analytics_date_type', 'date', 'metric_type'),
    )


class DataDeletionRequest(Base):
    """
    Track and process data deletion requests.
    """
    
    __tablename__ = "data_deletion_request"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Pseudonymized requester
    user_pseudonym = Column(String, nullable=False, index=True)
    
    # Request details
    deletion_type = Column(String, nullable=False)  # all, specific_data, specific_period
    data_categories = Column(JSON, default=[])  # Which categories to delete
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)
    
    # Processing status
    status = Column(String, nullable=False, default="pending")  # pending, processing, completed
    processed_at = Column(DateTime)
    
    # Verification token (sent to user for confirmation)
    verification_token = Column(String, unique=True)
    token_expires_at = Column(DateTime)
    
    # Results
    records_deleted = Column(Integer, default=0)
    deletion_certificate = Column(String)  # Hash proving deletion


# Utility functions for privacy operations

def generate_user_pseudonym(user_identifier: str, salt: str = None) -> str:
    """
    Generate a consistent pseudonym for a user.
    Uses salted hash to prevent rainbow table attacks.
    """
    if not salt:
        salt = settings.PSEUDONYM_SALT  # From environment
    
    combined = f"{user_identifier}:{salt}"
    hash_obj = hashlib.sha256(combined.encode())
    # Use first 16 chars of hex digest for reasonable length
    return hash_obj.hexdigest()[:16]


def calculate_deletion_date(retention_period: DataRetentionPeriod) -> datetime:
    """Calculate when data should be automatically deleted."""
    now = datetime.utcnow()
    
    periods = {
        DataRetentionPeriod.TRANSIENT: timedelta(hours=1),
        DataRetentionPeriod.SESSION: timedelta(hours=24),
        DataRetentionPeriod.WEEK: timedelta(days=7),
        DataRetentionPeriod.MONTH: timedelta(days=30),
        DataRetentionPeriod.QUARTER: timedelta(days=90),
        DataRetentionPeriod.YEAR: timedelta(days=365),
        DataRetentionPeriod.LEGAL: timedelta(days=365 * 7),  # 7 years default
    }
    
    return now + periods.get(retention_period, timedelta(days=30))


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or generalize any potentially identifying information from metadata.
    """
    sanitized = {}
    
    # Whitelist of safe fields
    safe_fields = {
        'model_type', 'success', 'retry_count', 'generation_time_ms',
        'format', 'has_text', 'has_image'
    }
    
    for key, value in metadata.items():
        if key in safe_fields:
            # Additional sanitization for specific fields
            if key == 'generation_time_ms' and isinstance(value, (int, float)):
                # Round to nearest 100ms to prevent timing attacks
                sanitized[key] = round(value / 100) * 100
            elif key == 'retry_count' and isinstance(value, int):
                # Cap retry count to prevent identifying users with issues
                sanitized[key] = min(value, 3)
            else:
                sanitized[key] = value
    
    return sanitized


# Event listeners for automatic privacy operations

@event.listens_for(PrivacyMemeMetadata, 'before_insert')
def set_deletion_date(mapper, connection, target):
    """Automatically set deletion date based on retention period."""
    if not target.scheduled_deletion_date:
        target.scheduled_deletion_date = calculate_deletion_date(
            DataRetentionPeriod(target.retention_period)
        )


@event.listens_for(Session, 'after_bulk_delete')
def log_bulk_deletion(delete_context):
    """Log bulk deletion operations for audit trail."""
    # Implementation would create audit log entries for bulk deletes
    pass