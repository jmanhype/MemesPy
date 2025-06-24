"""Privacy service for GDPR-compliant data handling."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid import uuid4
import hashlib
import secrets
import logging
from sqlalchemy import select, delete, update, and_, or_, func, exists
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.db_models.privacy_metadata import (
    PrivacyMemeMetadata,
    UserConsent,
    PrivacyAuditLog,
    AnonymizedAnalytics,
    DataDeletionRequest,
    ConsentType,
    DataRetentionPeriod,
    generate_user_pseudonym,
    calculate_deletion_date,
    sanitize_metadata,
)
from ..config.config import settings
from ..exceptions.specific import PrivacyError, ConsentError

logger = logging.getLogger(__name__)


class PrivacyService:
    """
    Service for managing privacy, consent, and GDPR compliance.

    Implements:
    - Consent management
    - Data minimization
    - Automatic deletion
    - Right to erasure
    - Data portability
    - Audit logging
    """

    def __init__(self):
        # Get encryption key from settings, with fallback
        try:
            self.encryption_key = settings.encryption_key
            self.pseudonym_salt = settings.pseudonym_salt
        except AttributeError:
            # Fallback if settings don't have these attributes
            import os

            self.encryption_key = os.environ.get("ENCRYPTION_KEY")
            self.pseudonym_salt = os.environ.get("PSEUDONYM_SALT")
            if not self.encryption_key or not self.pseudonym_salt:
                raise ValueError(
                    "ENCRYPTION_KEY and PSEUDONYM_SALT environment variables must be set for privacy service"
                )
            logger.warning(
                "Using environment variables for encryption settings. For production, configure these in settings."
            )

        self.min_k_anonymity = 5

    async def check_consent(
        self, session: AsyncSession, user_identifier: str, consent_type: ConsentType
    ) -> bool:
        """Check if user has given valid consent for specific purpose."""
        user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

        stmt = select(UserConsent).where(
            and_(
                UserConsent.user_pseudonym == user_pseudonym,
                UserConsent.consent_type == consent_type,
                UserConsent.granted == True,
                or_(UserConsent.revoked_at.is_(None), UserConsent.revoked_at > datetime.utcnow()),
                or_(UserConsent.expires_at.is_(None), UserConsent.expires_at > datetime.utcnow()),
            )
        )

        result = await session.execute(stmt)
        consent = result.scalar_one_or_none()

        # Log consent check
        await self._log_audit(
            session,
            action_type="consent_check",
            resource_type="consent",
            user_pseudonym=user_pseudonym,
            purpose=f"Check {consent_type} consent",
        )

        return consent is not None

    async def grant_consent(
        self,
        session: AsyncSession,
        user_identifier: str,
        consent_types: List[ConsentType],
        ip_country: Optional[str] = None,
        duration_days: Optional[int] = 365,
    ) -> Dict[str, Any]:
        """Grant consent for specified purposes."""
        user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

        granted_consents = []

        for consent_type in consent_types:
            # Check if consent already exists
            stmt = select(UserConsent).where(
                and_(
                    UserConsent.user_pseudonym == user_pseudonym,
                    UserConsent.consent_type == consent_type,
                )
            )
            result = await session.execute(stmt)
            consent = result.scalar_one_or_none()

            if consent:
                # Update existing consent
                consent.granted = True
                consent.granted_at = datetime.utcnow()
                consent.revoked_at = None
                consent.expires_at = (
                    datetime.utcnow() + timedelta(days=duration_days) if duration_days else None
                )
                consent.version = settings.privacy_policy_version
                consent.ip_country = ip_country
            else:
                # Create new consent
                consent = UserConsent(
                    user_pseudonym=user_pseudonym,
                    consent_type=consent_type,
                    granted=True,
                    granted_at=datetime.utcnow(),
                    expires_at=(
                        datetime.utcnow() + timedelta(days=duration_days) if duration_days else None
                    ),
                    version=settings.privacy_policy_version,
                    ip_country=ip_country,
                    purpose=self._get_consent_purpose(consent_type),
                    legal_basis="consent",
                )
                session.add(consent)

            granted_consents.append(consent_type)

        await session.commit()

        # Log consent grant
        await self._log_audit(
            session,
            action_type="consent_grant",
            resource_type="consent",
            user_pseudonym=user_pseudonym,
            purpose=f"Grant consent for {', '.join(granted_consents)}",
        )

        return {
            "status": "granted",
            "consents": granted_consents,
            "expires_at": (
                (datetime.utcnow() + timedelta(days=duration_days)).isoformat()
                if duration_days
                else None
            ),
            "version": settings.privacy_policy_version,
        }

    async def revoke_consent(
        self, session: AsyncSession, user_identifier: str, consent_types: List[ConsentType]
    ) -> Dict[str, Any]:
        """Revoke consent for specified purposes."""
        user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

        revoked_consents = []

        for consent_type in consent_types:
            stmt = select(UserConsent).where(
                and_(
                    UserConsent.user_pseudonym == user_pseudonym,
                    UserConsent.consent_type == consent_type,
                )
            )
            result = await session.execute(stmt)
            consent = result.scalar_one_or_none()

            if consent and consent.granted:
                consent.granted = False
                consent.revoked_at = datetime.utcnow()
                revoked_consents.append(consent_type)

        await session.commit()

        # Log consent revocation
        await self._log_audit(
            session,
            action_type="consent_revoke",
            resource_type="consent",
            user_pseudonym=user_pseudonym,
            purpose=f"Revoke consent for {', '.join(revoked_consents)}",
        )

        # Handle data deletion for revoked consents
        if ConsentType.ANALYTICS in revoked_consents:
            await self._delete_analytics_data(session, user_pseudonym)

        if ConsentType.PERSONALIZATION in revoked_consents:
            await self._delete_personalization_data(session, user_pseudonym)

        return {"status": "revoked", "consents": revoked_consents, "data_deleted": True}

    async def create_privacy_safe_meme(
        self,
        session: AsyncSession,
        user_identifier: Optional[str],
        meme_data: Dict[str, Any],
        retention_period: DataRetentionPeriod = DataRetentionPeriod.MONTH,
    ) -> PrivacyMemeMetadata:
        """Create a meme with privacy-safe metadata."""
        # Generate pseudonym if user provided
        user_pseudonym = None
        if user_identifier:
            user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

            # Check essential consent
            has_consent = await self.check_consent(session, user_identifier, ConsentType.ESSENTIAL)
            if not has_consent:
                raise ConsentError("Essential consent required for meme generation")

        # Sanitize all metadata
        safe_generation_metadata = sanitize_metadata(meme_data.get("generation_metadata", {}))

        # Create privacy-safe meme record
        meme = PrivacyMemeMetadata(
            id=str(uuid4()),
            user_pseudonym=user_pseudonym,
            session_id=meme_data.get("session_id"),
            topic=self._sanitize_topic(meme_data["topic"]),
            format=meme_data["format"],
            text=meme_data["text"],
            image_url=meme_data["image_url"],
            score=meme_data.get("score"),
            generation_metadata=safe_generation_metadata,
            safety_flags={
                "is_safe": meme_data.get("is_safe", True),
                "requires_moderation": meme_data.get("requires_moderation", False),
            },
            retention_period=retention_period,
            scheduled_deletion_date=calculate_deletion_date(retention_period),
        )

        session.add(meme)
        await session.commit()

        # Log creation
        await self._log_audit(
            session,
            action_type="create",
            resource_type="meme",
            resource_id=meme.id,
            user_pseudonym=user_pseudonym,
            purpose="Meme generation",
        )

        return meme

    async def request_data_deletion(
        self,
        session: AsyncSession,
        user_identifier: str,
        deletion_type: str = "all",
        data_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process right to erasure request."""
        user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

        # Create deletion request
        verification_token = secrets.token_urlsafe(32)

        request = DataDeletionRequest(
            user_pseudonym=user_pseudonym,
            deletion_type=deletion_type,
            data_categories=data_categories or [],
            verification_token=verification_token,
            token_expires_at=datetime.utcnow() + timedelta(hours=24),
        )

        session.add(request)
        await session.commit()

        # Log deletion request
        await self._log_audit(
            session,
            action_type="deletion_request",
            resource_type="user_data",
            user_pseudonym=user_pseudonym,
            purpose="Right to erasure request",
        )

        return {
            "request_id": request.id,
            "verification_token": verification_token,
            "status": "pending",
            "expires_at": request.token_expires_at.isoformat(),
        }

    async def execute_data_deletion(
        self, session: AsyncSession, request_id: str, verification_token: str
    ) -> Dict[str, Any]:
        """Execute verified data deletion request."""
        # Verify request
        stmt = select(DataDeletionRequest).where(
            and_(
                DataDeletionRequest.id == request_id,
                DataDeletionRequest.verification_token == verification_token,
                DataDeletionRequest.status == "pending",
                DataDeletionRequest.token_expires_at > datetime.utcnow(),
            )
        )

        result = await session.execute(stmt)
        request = result.scalar_one_or_none()

        if not request:
            raise PrivacyError("Invalid or expired deletion request")

        # Update request status
        request.status = "processing"
        await session.commit()

        # Delete data based on request type
        deleted_counts = {}

        if request.deletion_type == "all":
            # Delete all user data
            deleted_counts["memes"] = await self._delete_user_memes(session, request.user_pseudonym)
            deleted_counts["consents"] = await self._delete_user_consents(
                session, request.user_pseudonym
            )
            deleted_counts["audit_logs"] = await self._delete_user_audit_logs(
                session, request.user_pseudonym
            )

        # Generate deletion certificate
        deletion_data = f"{request_id}:{deleted_counts}:{datetime.utcnow().isoformat()}"
        certificate = hashlib.sha256(deletion_data.encode()).hexdigest()

        # Update request
        request.status = "completed"
        request.processed_at = datetime.utcnow()
        request.records_deleted = sum(deleted_counts.values())
        request.deletion_certificate = certificate

        await session.commit()

        return {
            "request_id": request_id,
            "status": "completed",
            "records_deleted": deleted_counts,
            "certificate": certificate,
        }

    async def export_user_data(self, session: AsyncSession, user_identifier: str) -> Dict[str, Any]:
        """Export all user data for portability."""
        user_pseudonym = generate_user_pseudonym(user_identifier, self.pseudonym_salt)

        # Check consent for data export
        has_consent = await self.check_consent(session, user_identifier, ConsentType.ESSENTIAL)
        if not has_consent:
            raise ConsentError("Consent required for data export")

        # Collect all user data
        data = {
            "export_date": datetime.utcnow().isoformat(),
            "user_pseudonym": user_pseudonym,
            "data_categories": {},
        }

        # Memes
        memes_stmt = select(PrivacyMemeMetadata).where(
            PrivacyMemeMetadata.user_pseudonym == user_pseudonym
        )
        memes_result = await session.execute(memes_stmt)
        memes = memes_result.scalars().all()

        data["data_categories"]["memes"] = [meme.anonymize() for meme in memes]

        # Consents
        consents_stmt = select(UserConsent).where(UserConsent.user_pseudonym == user_pseudonym)
        consents_result = await session.execute(consents_stmt)
        consents = consents_result.scalars().all()

        data["data_categories"]["consents"] = [
            {
                "type": consent.consent_type,
                "granted": consent.granted,
                "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
                "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
            }
            for consent in consents
        ]

        # Log export
        await self._log_audit(
            session,
            action_type="export",
            resource_type="user_data",
            user_pseudonym=user_pseudonym,
            purpose="Data portability request",
        )

        return data

    async def run_automatic_deletion(
        self, session: AsyncSession, batch_size: int = 100
    ) -> Dict[str, int]:
        """Run automatic deletion of expired data."""
        deleted_counts = {"memes": 0, "audit_logs": 0}

        # Delete expired memes
        while True:
            stmt = (
                select(PrivacyMemeMetadata)
                .where(PrivacyMemeMetadata.scheduled_deletion_date <= datetime.utcnow())
                .limit(batch_size)
            )

            result = await session.execute(stmt)
            memes = result.scalars().all()

            if not memes:
                break

            for meme in memes:
                await session.delete(meme)
                deleted_counts["memes"] += 1

            await session.commit()

            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.1)

        # Delete old audit logs (keep for legal minimum)
        cutoff_date = datetime.utcnow() - timedelta(days=365 * 7)  # 7 years

        stmt = delete(PrivacyAuditLog).where(PrivacyAuditLog.timestamp < cutoff_date)

        result = await session.execute(stmt)
        deleted_counts["audit_logs"] = result.rowcount
        await session.commit()

        logger.info(f"Automatic deletion completed: {deleted_counts}")

        return deleted_counts

    async def generate_analytics(
        self, session: AsyncSession, date_range: tuple[datetime, datetime], dimensions: List[str]
    ) -> List[AnonymizedAnalytics]:
        """Generate anonymized analytics with k-anonymity."""
        start_date, end_date = date_range

        analytics = []

        for dimension in dimensions:
            # Example: Generate format-based analytics
            if dimension == "format":
                stmt = (
                    select(
                        PrivacyMemeMetadata.format,
                        func.count(PrivacyMemeMetadata.id).label("count"),
                        func.avg(PrivacyMemeMetadata.score).label("avg_score"),
                    )
                    .where(
                        and_(
                            PrivacyMemeMetadata.created_at >= start_date,
                            PrivacyMemeMetadata.created_at < end_date,
                        )
                    )
                    .group_by(PrivacyMemeMetadata.format)
                )

                result = await session.execute(stmt)

                for row in result:
                    # Only include if k-anonymity threshold met
                    if row.count >= self.min_k_anonymity:
                        analytics.append(
                            AnonymizedAnalytics(
                                date=start_date,
                                granularity="daily",
                                metric_type="meme_count",
                                dimension=f"format:{row.format}",
                                count=row.count,
                                avg_value=float(row.avg_score) if row.avg_score else None,
                                k_value=row.count,
                            )
                        )

        # Save analytics
        session.add_all(analytics)
        await session.commit()

        return analytics

    # Private helper methods

    async def _log_audit(
        self,
        session: AsyncSession,
        action_type: str,
        resource_type: str,
        user_pseudonym: Optional[str] = None,
        resource_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ):
        """Create privacy-safe audit log entry."""
        audit = PrivacyAuditLog(
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            user_pseudonym=user_pseudonym,
            system_component="privacy_service",
            purpose=purpose,
            legal_basis="legitimate_interest",
            country_code=None,  # Would be set from request context
            hour_of_day=datetime.utcnow().hour,
        )

        session.add(audit)
        # Don't commit here - let caller handle transaction

    def _sanitize_topic(self, topic: str) -> str:
        """Remove any potential PII from topic."""
        # Remove email addresses, phone numbers, etc.
        # This is a simplified example
        import re

        # Remove email addresses
        topic = re.sub(r"\S+@\S+", "[email]", topic)

        # Remove phone numbers
        topic = re.sub(r"\b\d{10,}\b", "[phone]", topic)

        # Remove potential names (simplified)
        # In production, use a proper NER system

        return topic.strip()

    def _get_consent_purpose(self, consent_type: ConsentType) -> str:
        """Get human-readable purpose for consent type."""
        purposes = {
            ConsentType.ESSENTIAL: "Essential service operation and meme generation",
            ConsentType.ANALYTICS: "Anonymous usage analytics and service improvement",
            ConsentType.PERSONALIZATION: "Personalized recommendations and preferences",
            ConsentType.MARKETING: "Marketing communications and updates",
        }
        return purposes.get(consent_type, "Service usage")

    async def _delete_user_memes(self, session: AsyncSession, user_pseudonym: str) -> int:
        """Delete all memes for a user."""
        stmt = delete(PrivacyMemeMetadata).where(
            PrivacyMemeMetadata.user_pseudonym == user_pseudonym
        )
        result = await session.execute(stmt)
        return result.rowcount

    async def _delete_user_consents(self, session: AsyncSession, user_pseudonym: str) -> int:
        """Delete all consents for a user."""
        stmt = delete(UserConsent).where(UserConsent.user_pseudonym == user_pseudonym)
        result = await session.execute(stmt)
        return result.rowcount

    async def _delete_user_audit_logs(self, session: AsyncSession, user_pseudonym: str) -> int:
        """Delete audit logs for a user (where legally allowed)."""
        # Keep audit logs for legal minimum period
        cutoff_date = datetime.utcnow() - timedelta(days=365)  # 1 year minimum

        stmt = delete(PrivacyAuditLog).where(
            and_(
                PrivacyAuditLog.user_pseudonym == user_pseudonym,
                PrivacyAuditLog.timestamp < cutoff_date,
            )
        )
        result = await session.execute(stmt)
        return result.rowcount

    async def _delete_analytics_data(self, session: AsyncSession, user_pseudonym: str):
        """Remove user from analytics tracking."""
        # Mark user as opted out of analytics
        # Future analytics will exclude this pseudonym
        pass

    async def _delete_personalization_data(self, session: AsyncSession, user_pseudonym: str):
        """Delete personalization data for user."""
        # Delete user preferences, recommendations, etc.
        pass
