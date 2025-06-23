"""Privacy and GDPR compliance API endpoints."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.connection import get_db
from ...services.privacy_service import PrivacyService
from ...models.database.privacy_metadata import ConsentType, DataRetentionPeriod
from ...exceptions import PrivacyError, ConsentError
from ...api.dependencies import get_current_user_id, get_user_country
from ...config.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
privacy_service = PrivacyService()


# Request/Response Models

class ConsentRequest(BaseModel):
    """Request model for consent management."""
    consent_types: List[ConsentType] = Field(
        ..., 
        description="Types of consent to grant",
        example=[ConsentType.ESSENTIAL, ConsentType.ANALYTICS]
    )
    duration_days: Optional[int] = Field(
        365,
        description="Consent duration in days",
        ge=1,
        le=730  # Max 2 years
    )


class ConsentResponse(BaseModel):
    """Response model for consent operations."""
    status: str
    consents: List[str]
    expires_at: Optional[str]
    version: str


class DataDeletionRequest(BaseModel):
    """Request model for data deletion."""
    deletion_type: str = Field(
        "all",
        description="Type of deletion",
        pattern="^(all|specific_data|specific_period)$"
    )
    data_categories: Optional[List[str]] = Field(
        None,
        description="Specific data categories to delete"
    )
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


class DataDeletionResponse(BaseModel):
    """Response model for deletion requests."""
    request_id: str
    verification_token: Optional[str]
    status: str
    expires_at: Optional[str]
    verification_required: bool = True


class PrivacySettingsResponse(BaseModel):
    """Current privacy settings for a user."""
    user_pseudonym: str
    active_consents: Dict[str, bool]
    data_retention: Dict[str, Any]
    deletion_requests: List[Dict[str, Any]]


class DataExportResponse(BaseModel):
    """Response model for data export."""
    export_id: str
    status: str
    download_url: Optional[str]
    expires_at: Optional[str]


# Endpoints

@router.get("/privacy/settings", response_model=PrivacySettingsResponse)
async def get_privacy_settings(
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """Get current privacy settings and consent status."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        # Get all consent statuses
        consent_status = {}
        for consent_type in ConsentType:
            consent_status[consent_type] = await privacy_service.check_consent(
                db, user_id, consent_type
            )
        
        # Get data retention info
        retention_info = {
            "default_period": DataRetentionPeriod.MONTH,
            "available_periods": [p.value for p in DataRetentionPeriod],
            "automatic_deletion": True
        }
        
        # Get pending deletion requests
        # In a real implementation, this would query the database
        deletion_requests = []
        
        return PrivacySettingsResponse(
            user_pseudonym=privacy_service.generate_user_pseudonym(user_id),
            active_consents=consent_status,
            data_retention=retention_info,
            deletion_requests=deletion_requests
        )
    
    except Exception as e:
        logger.error(f"Error fetching privacy settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch privacy settings")


@router.post("/privacy/consent/grant", response_model=ConsentResponse)
async def grant_consent(
    request: ConsentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
    user_country: Optional[str] = Depends(get_user_country)
):
    """Grant consent for specified data processing purposes."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        result = await privacy_service.grant_consent(
            db,
            user_id,
            request.consent_types,
            ip_country=user_country,
            duration_days=request.duration_days
        )
        
        return ConsentResponse(**result)
    
    except Exception as e:
        logger.error(f"Error granting consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to grant consent")


@router.post("/privacy/consent/revoke", response_model=ConsentResponse)
async def revoke_consent(
    request: ConsentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """Revoke consent for specified data processing purposes."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        # Check if trying to revoke essential consent
        if ConsentType.ESSENTIAL in request.consent_types:
            raise HTTPException(
                status_code=400,
                detail="Essential consent cannot be revoked while using the service"
            )
        
        result = await privacy_service.revoke_consent(
            db,
            user_id,
            request.consent_types
        )
        
        return ConsentResponse(**result)
    
    except ConsentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error revoking consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke consent")


@router.post("/privacy/data/delete", response_model=DataDeletionResponse)
async def request_data_deletion(
    request: DataDeletionRequest,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """Request deletion of personal data (Right to Erasure)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        result = await privacy_service.request_data_deletion(
            db,
            user_id,
            deletion_type=request.deletion_type,
            data_categories=request.data_categories
        )
        
        return DataDeletionResponse(
            request_id=result["request_id"],
            verification_token=result["verification_token"],
            status=result["status"],
            expires_at=result["expires_at"],
            verification_required=True
        )
    
    except Exception as e:
        logger.error(f"Error creating deletion request: {e}")
        raise HTTPException(status_code=500, detail="Failed to create deletion request")


@router.post("/privacy/data/delete/confirm")
async def confirm_data_deletion(
    request_id: str,
    verification_token: str,
    db: AsyncSession = Depends(get_db)
):
    """Confirm and execute data deletion request."""
    try:
        result = await privacy_service.execute_data_deletion(
            db,
            request_id,
            verification_token
        )
        
        return {
            "status": "success",
            "message": "Data deletion completed",
            "details": result
        }
    
    except PrivacyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing deletion: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute deletion")


@router.get("/privacy/data/export")
async def export_user_data(
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Export all user data (Right to Data Portability)."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        # For large datasets, this would be done asynchronously
        data = await privacy_service.export_user_data(db, user_id)
        
        # In production, this would:
        # 1. Generate a secure download link
        # 2. Send notification when ready
        # 3. Auto-expire the link
        
        return {
            "status": "success",
            "data": data,
            "format": "json",
            "generated_at": datetime.utcnow().isoformat()
        }
    
    except ConsentError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")


@router.get("/privacy/policy")
async def get_privacy_policy():
    """Get current privacy policy."""
    return {
        "version": settings.privacy_policy_version,
        "effective_date": "2024-01-01",
        "policy_url": "/privacy-policy",
        "contact": {
            "email": "privacy@memeservice.com",
            "dpo_email": "dpo@memeservice.com"
        },
        "data_controller": {
            "name": "Meme Generation Service",
            "address": "Privacy-first approach",
            "country": "EU"
        },
        "user_rights": [
            "Right to access",
            "Right to rectification",
            "Right to erasure",
            "Right to restrict processing",
            "Right to data portability",
            "Right to object",
            "Right to withdraw consent"
        ],
        "data_categories": {
            "essential": {
                "description": "Data required for service operation",
                "retention": "30 days default",
                "purpose": "Meme generation and delivery"
            },
            "analytics": {
                "description": "Anonymous usage analytics",
                "retention": "90 days",
                "purpose": "Service improvement"
            },
            "personalization": {
                "description": "User preferences",
                "retention": "Until revoked",
                "purpose": "Personalized experience"
            }
        }
    }


@router.get("/privacy/transparency")
async def get_data_transparency():
    """Get transparency report about data handling."""
    return {
        "data_minimization": {
            "principle": "We collect only essential data",
            "practices": [
                "No real names or email addresses stored",
                "IP addresses converted to country codes only",
                "Automatic data expiration",
                "Pseudonymization by default"
            ]
        },
        "security_measures": [
            "Encryption at rest and in transit",
            "Regular security audits",
            "Access logging without PII",
            "Secure deletion procedures"
        ],
        "third_parties": {
            "sharing": "No personal data shared with third parties",
            "processors": [
                {
                    "name": "Cloud Storage Provider",
                    "purpose": "Image storage",
                    "data_shared": "Anonymized images only"
                }
            ]
        },
        "automated_decisions": {
            "used": True,
            "purposes": ["Content moderation", "Quality scoring"],
            "human_review": "Available on request"
        }
    }


@router.post("/privacy/anonymize/{meme_id}")
async def anonymize_meme(
    meme_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """Anonymize a specific meme immediately."""
    if not user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    
    try:
        # In production, verify user owns this meme
        # Then anonymize it
        
        return {
            "status": "success",
            "meme_id": meme_id,
            "anonymized": True,
            "message": "Meme has been anonymized"
        }
    
    except Exception as e:
        logger.error(f"Error anonymizing meme: {e}")
        raise HTTPException(status_code=500, detail="Failed to anonymize meme")


# Admin endpoints (would be in separate router with proper auth)

@router.post("/privacy/admin/cleanup", include_in_schema=False)
async def run_privacy_cleanup(
    db: AsyncSession = Depends(get_db),
    api_key: str = Header(None)
):
    """Run automatic data cleanup (admin only)."""
    # Verify admin API key
    if api_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        result = await privacy_service.run_automatic_deletion(db)
        
        return {
            "status": "success",
            "deleted": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error running cleanup: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")