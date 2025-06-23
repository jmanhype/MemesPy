# Privacy-First Metadata Collection System

## Overview

This document describes the privacy-first redesign of the metadata collection system for the DSPy Meme Generator, implementing full GDPR compliance and user privacy protection.

## Core Principles

### 1. Data Minimization
- **Only Essential Data**: We collect only the minimum data necessary for service operation
- **No PII Storage**: Personal Identifiable Information is never stored directly
- **Generalization**: Specific data points are generalized into categories (e.g., exact timestamps → hour of day)
- **Sanitization**: All user inputs are sanitized to remove potential PII

### 2. Privacy by Design
- **Default Protection**: Privacy controls are enabled by default
- **Pseudonymization**: User identifiers are one-way hashed with salts
- **Automatic Expiration**: All data has automatic deletion dates
- **Secure Deletion**: Data is securely deleted when no longer needed

### 3. User Control
- **Granular Consent**: Users control consent for different data purposes
- **Data Portability**: Users can export all their data
- **Right to Erasure**: Users can request immediate deletion
- **Transparency**: Clear information about data collection and use

## Architecture Components

### Database Models

#### PrivacyMemeMetadata
- Stores minimal meme data with automatic deletion scheduling
- No direct user identifiers, only pseudonyms
- Sanitized metadata with no PII

#### UserConsent
- Explicit consent tracking with expiration
- Granular consent types (essential, analytics, personalization, marketing)
- Full audit trail of consent changes

#### PrivacyAuditLog
- Anonymized audit logging without PII
- Only stores generalized information (country, hour of day)
- Automatic cleanup based on legal requirements

#### AnonymizedAnalytics
- Aggregated analytics with k-anonymity (minimum k=5)
- No individual user tracking
- Only statistical aggregates stored

### Privacy Service

The `PrivacyService` class handles all privacy operations:

```python
# Check consent
has_consent = await privacy_service.check_consent(
    session, user_id, ConsentType.ANALYTICS
)

# Grant consent
result = await privacy_service.grant_consent(
    session, user_id, [ConsentType.ESSENTIAL, ConsentType.ANALYTICS]
)

# Request data deletion
deletion_request = await privacy_service.request_data_deletion(
    session, user_id, deletion_type="all"
)

# Export user data
user_data = await privacy_service.export_user_data(session, user_id)
```

### Privacy Middleware

Two middleware components enforce privacy:

1. **PrivacyMiddleware**: 
   - Adds privacy headers to all responses
   - Sanitizes request/response data
   - Implements request anonymization

2. **ConsentEnforcementMiddleware**:
   - Enforces consent requirements for endpoints
   - Blocks access without proper consent

### Scheduled Tasks

Automated privacy tasks run on schedule:

- **Hourly**: Automatic deletion of expired data
- **Daily**: Analytics aggregation with k-anonymity
- **Daily**: Consent expiration checks
- **Weekly**: Audit log cleanup
- **Daily**: Data integrity verification

## API Endpoints

### Privacy Management
- `GET /api/privacy/settings` - Get current privacy settings
- `POST /api/privacy/consent/grant` - Grant consent
- `POST /api/privacy/consent/revoke` - Revoke consent
- `GET /api/privacy/policy` - Get privacy policy

### Data Control
- `POST /api/privacy/data/delete` - Request data deletion
- `POST /api/privacy/data/delete/confirm` - Confirm deletion
- `GET /api/privacy/data/export` - Export user data
- `POST /api/privacy/anonymize/{meme_id}` - Anonymize specific meme

## Data Retention Policies

| Data Type | Default Retention | Purpose |
|-----------|------------------|---------|
| Transient | 1 hour | Temporary processing |
| Session | 24 hours | Session data |
| Weekly | 7 days | Short-term features |
| Monthly | 30 days | Standard meme data |
| Quarterly | 90 days | Analytics data |
| Annual | 365 days | Long-term trends |
| Legal | 7 years | Legal requirements |

## Security Measures

### Encryption
- All sensitive data encrypted at rest
- TLS for data in transit
- Secure key management

### Access Control
- Role-based access control
- API key authentication
- Audit logging of all access

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF tokens

## Compliance Features

### GDPR Rights Implementation
1. **Right to Access**: Data export functionality
2. **Right to Rectification**: Update mechanisms
3. **Right to Erasure**: Delete functionality
4. **Right to Restrict Processing**: Consent management
5. **Right to Data Portability**: JSON export
6. **Right to Object**: Opt-out mechanisms

### Consent Management
- Explicit consent required
- Granular control by purpose
- Easy withdrawal mechanism
- Consent version tracking

### Data Processing Records
- Automated audit logging
- Processing purpose tracking
- Legal basis documentation
- Data flow mapping

## Best Practices

### For Developers
1. Never store PII directly
2. Always use pseudonyms for user references
3. Implement data minimization in new features
4. Add automatic deletion for new data types
5. Document data collection purposes

### For Operations
1. Regular privacy audits
2. Monitor automatic deletion jobs
3. Review access logs
4. Update privacy policy versions
5. Respond to user requests promptly

## Monitoring and Alerts

The system includes monitoring for:
- Failed deletion jobs
- Low k-anonymity in analytics
- Expired data not deleted
- PII detection in anonymized fields
- Consent expiration

## Future Enhancements

1. **Differential Privacy**: Add noise to analytics
2. **Homomorphic Encryption**: Process encrypted data
3. **Federated Learning**: Train models without data collection
4. **Zero-Knowledge Proofs**: Verify without revealing data
5. **Privacy Budget**: Track cumulative privacy loss