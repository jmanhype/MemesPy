"""Scheduled tasks for privacy and data management."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete, update, and_, func, exists
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ..services.privacy_service import PrivacyService
from ..models.db_models.privacy_metadata import (
    PrivacyMemeMetadata,
    PrivacyAuditLog,
    AnonymizedAnalytics,
    UserConsent,
)
from ..config.config import settings
from ..models.connection import db_manager

logger = logging.getLogger(__name__)


class PrivacyTaskScheduler:
    """
    Manages scheduled privacy-related tasks.

    Tasks include:
    - Automatic data deletion
    - Analytics aggregation
    - Consent expiration checks
    - Audit log cleanup
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.privacy_service = PrivacyService()
        self.engine = db_manager.async_engine
        self.AsyncSessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    def start(self):
        """Start all scheduled privacy tasks."""
        # Schedule automatic deletion - runs every hour
        self.scheduler.add_job(
            self.run_automatic_deletion,
            trigger=IntervalTrigger(hours=1),
            id="auto_deletion",
            name="Automatic data deletion",
            misfire_grace_time=300,  # 5 minutes grace period
        )

        # Schedule analytics aggregation - runs daily at 2 AM
        self.scheduler.add_job(
            self.aggregate_analytics,
            trigger=CronTrigger(hour=2, minute=0),
            id="analytics_aggregation",
            name="Daily analytics aggregation",
        )

        # Schedule consent expiration check - runs daily at 3 AM
        self.scheduler.add_job(
            self.check_consent_expirations,
            trigger=CronTrigger(hour=3, minute=0),
            id="consent_expiration",
            name="Check consent expirations",
        )

        # Schedule audit log cleanup - runs weekly on Sunday at 4 AM
        self.scheduler.add_job(
            self.cleanup_audit_logs,
            trigger=CronTrigger(day_of_week=0, hour=4, minute=0),
            id="audit_cleanup",
            name="Weekly audit log cleanup",
        )

        # Schedule data integrity check - runs daily at 5 AM
        self.scheduler.add_job(
            self.verify_data_integrity,
            trigger=CronTrigger(hour=5, minute=0),
            id="data_integrity",
            name="Daily data integrity check",
        )

        # Start the scheduler
        self.scheduler.start()
        logger.info("Privacy task scheduler started")

    def stop(self):
        """Stop all scheduled tasks."""
        self.scheduler.shutdown()
        logger.info("Privacy task scheduler stopped")

    async def run_automatic_deletion(self):
        """Execute automatic deletion of expired data."""
        logger.info("Starting automatic data deletion task")

        async with self.AsyncSessionLocal() as session:
            try:
                # Delete expired memes
                deleted_counts = await self.privacy_service.run_automatic_deletion(
                    session, batch_size=100
                )

                # Log results
                logger.info(f"Automatic deletion completed: {deleted_counts}")

                # Create audit log entry for the deletion
                audit = PrivacyAuditLog(
                    action_type="automatic_deletion",
                    resource_type="system",
                    system_component="privacy_scheduler",
                    purpose="Scheduled data retention enforcement",
                    legal_basis="data_minimization",
                )
                session.add(audit)
                await session.commit()

            except Exception as e:
                logger.error(f"Error in automatic deletion task: {e}")
                await session.rollback()

    async def aggregate_analytics(self):
        """Aggregate analytics data with k-anonymity."""
        logger.info("Starting analytics aggregation task")

        async with self.AsyncSessionLocal() as session:
            try:
                # Calculate date range for aggregation
                end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                start_date = end_date - timedelta(days=1)

                # Generate aggregated analytics
                analytics = await self.privacy_service.generate_analytics(
                    session,
                    date_range=(start_date, end_date),
                    dimensions=["format", "topic_category", "quality_tier"],
                )

                logger.info(f"Generated {len(analytics)} analytics records")

                # Clean up raw data that's been aggregated
                # (keeping raw data for minimum period for debugging)
                cleanup_date = datetime.utcnow() - timedelta(days=7)

                # Mark old detailed data for expedited deletion
                stmt = (
                    update(PrivacyMemeMetadata)
                    .where(
                        and_(
                            PrivacyMemeMetadata.created_at < cleanup_date,
                            PrivacyMemeMetadata.scheduled_deletion_date > datetime.utcnow(),
                        )
                    )
                    .values(scheduled_deletion_date=datetime.utcnow() + timedelta(hours=24))
                )

                await session.execute(stmt)
                await session.commit()

            except Exception as e:
                logger.error(f"Error in analytics aggregation: {e}")
                await session.rollback()

    async def check_consent_expirations(self):
        """Check and handle expired consents."""
        logger.info("Checking consent expirations")

        async with self.AsyncSessionLocal() as session:
            try:
                # Find expired consents
                stmt = select(UserConsent).where(
                    and_(
                        UserConsent.expires_at.isnot(None),
                        UserConsent.expires_at <= datetime.utcnow(),
                        UserConsent.granted == True,
                        UserConsent.revoked_at.is_(None),
                    )
                )

                result = await session.execute(stmt)
                expired_consents = result.scalars().all()

                # Revoke expired consents
                for consent in expired_consents:
                    consent.granted = False
                    consent.revoked_at = datetime.utcnow()

                    logger.info(
                        f"Auto-revoked expired consent: "
                        f"user={consent.user_pseudonym}, type={consent.consent_type}"
                    )

                await session.commit()

                logger.info(f"Processed {len(expired_consents)} expired consents")

            except Exception as e:
                logger.error(f"Error checking consent expirations: {e}")
                await session.rollback()

    async def cleanup_audit_logs(self):
        """Clean up old audit logs while preserving legal requirements."""
        logger.info("Starting audit log cleanup")

        async with self.AsyncSessionLocal() as session:
            try:
                # Different retention periods for different types
                retention_periods = {
                    "consent_grant": timedelta(days=365 * 3),  # 3 years
                    "consent_revoke": timedelta(days=365 * 3),  # 3 years
                    "deletion_request": timedelta(days=365 * 7),  # 7 years
                    "export": timedelta(days=365 * 2),  # 2 years
                    "create": timedelta(days=30),  # 30 days
                    "read": timedelta(days=7),  # 7 days
                }

                deleted_total = 0

                for action_type, retention_period in retention_periods.items():
                    cutoff_date = datetime.utcnow() - retention_period

                    stmt = delete(PrivacyAuditLog).where(
                        and_(
                            PrivacyAuditLog.action_type == action_type,
                            PrivacyAuditLog.timestamp < cutoff_date,
                        )
                    )

                    result = await session.execute(stmt)
                    deleted_count = result.rowcount
                    deleted_total += deleted_count

                    if deleted_count > 0:
                        logger.info(
                            f"Deleted {deleted_count} audit logs "
                            f"of type '{action_type}' older than {cutoff_date}"
                        )

                await session.commit()

                logger.info(f"Audit log cleanup completed. Total deleted: {deleted_total}")

            except Exception as e:
                logger.error(f"Error in audit log cleanup: {e}")
                await session.rollback()

    async def verify_data_integrity(self):
        """Verify data integrity and privacy compliance."""
        logger.info("Starting data integrity verification")

        issues_found = []

        async with self.AsyncSessionLocal() as session:
            try:
                # Check for orphaned records
                orphan_check = await self._check_orphaned_records(session)
                if orphan_check:
                    issues_found.extend(orphan_check)

                # Check for PII in fields that should be anonymized
                pii_check = await self._check_for_pii(session)
                if pii_check:
                    issues_found.extend(pii_check)

                # Check for records past deletion date
                expired_check = await self._check_expired_records(session)
                if expired_check:
                    issues_found.extend(expired_check)

                # Verify k-anonymity in analytics
                k_anonymity_check = await self._verify_k_anonymity(session)
                if k_anonymity_check:
                    issues_found.extend(k_anonymity_check)

                # Log results
                if issues_found:
                    logger.warning(
                        f"Data integrity issues found: {len(issues_found)} issues\n"
                        f"Issues: {issues_found}"
                    )

                    # Create alert (in production, would send to monitoring)
                    audit = PrivacyAuditLog(
                        action_type="integrity_check",
                        resource_type="system",
                        system_component="privacy_scheduler",
                        purpose=f"Found {len(issues_found)} integrity issues",
                        legal_basis="compliance_monitoring",
                    )
                    session.add(audit)
                else:
                    logger.info("Data integrity check passed - no issues found")

                await session.commit()

            except Exception as e:
                logger.error(f"Error in data integrity check: {e}")
                await session.rollback()

    async def _check_orphaned_records(self, session: AsyncSession) -> list:
        """Check for orphaned records that should have been deleted."""
        issues = []

        # Check for audit logs without corresponding memes
        stmt = select(PrivacyAuditLog).where(
            and_(
                PrivacyAuditLog.meme_id.isnot(None),
                ~exists().where(PrivacyMemeMetadata.id == PrivacyAuditLog.meme_id),
            )
        )

        result = await session.execute(stmt)
        orphaned_logs = result.scalars().all()

        if orphaned_logs:
            issues.append(
                {
                    "type": "orphaned_audit_logs",
                    "count": len(orphaned_logs),
                    "action": "will_delete",
                }
            )

            # Delete orphaned logs
            for log in orphaned_logs:
                await session.delete(log)

        return issues

    async def _check_for_pii(self, session: AsyncSession) -> list:
        """Check for potential PII in anonymized fields."""
        issues = []

        # Check for email patterns in topics
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

        stmt = select(PrivacyMemeMetadata).where(
            PrivacyMemeMetadata.topic.regexp_match(email_pattern)
        )

        result = await session.execute(stmt)
        memes_with_email = result.scalars().all()

        if memes_with_email:
            issues.append(
                {
                    "type": "pii_in_topic",
                    "count": len(memes_with_email),
                    "pattern": "email",
                    "action": "will_sanitize",
                }
            )

            # Sanitize the topics
            for meme in memes_with_email:
                meme.topic = self.privacy_service._sanitize_topic(meme.topic)

        return issues

    async def _check_expired_records(self, session: AsyncSession) -> list:
        """Check for records past their deletion date."""
        issues = []

        stmt = select(func.count(PrivacyMemeMetadata.id)).where(
            PrivacyMemeMetadata.scheduled_deletion_date <= datetime.utcnow()
        )

        result = await session.execute(stmt)
        expired_count = result.scalar()

        if expired_count > 0:
            issues.append(
                {
                    "type": "expired_records",
                    "count": expired_count,
                    "action": "will_delete_next_run",
                }
            )

        return issues

    async def _verify_k_anonymity(self, session: AsyncSession) -> list:
        """Verify k-anonymity in analytics data."""
        issues = []

        stmt = select(AnonymizedAnalytics).where(
            AnonymizedAnalytics.k_value < self.privacy_service.min_k_anonymity
        )

        result = await session.execute(stmt)
        low_k_records = result.scalars().all()

        if low_k_records:
            issues.append(
                {
                    "type": "low_k_anonymity",
                    "count": len(low_k_records),
                    "min_k": self.privacy_service.min_k_anonymity,
                    "action": "will_remove",
                }
            )

            # Remove records that don't meet k-anonymity
            for record in low_k_records:
                await session.delete(record)

        return issues


# Create global scheduler instance
privacy_scheduler = PrivacyTaskScheduler()


def start_privacy_tasks():
    """Start privacy task scheduler."""
    privacy_scheduler.start()


def stop_privacy_tasks():
    """Stop privacy task scheduler."""
    privacy_scheduler.stop()
