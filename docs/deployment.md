# Deployment Guide

This guide covers deploying the DSPy Meme Generation pipeline in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Platform Deployments](#cloud-platform-deployments)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying, ensure you have:

1. Application dependencies:
   - Python 3.9+
   - PostgreSQL 13+
   - Redis 6+
   - Docker and Docker Compose (for containerized deployment)
   - Kubernetes cluster (for orchestrated deployment)
   - OpenAI API key
   - Cloudinary account
   - Domain name and SSL certificate

2. Infrastructure tools:
   - Docker
   - kubectl (for Kubernetes)
   - Cloud provider CLI tools

3. Configuration files:
   - `.env.production`
   - SSL certificates
   - API keys and secrets

## Environment Setup

### Production Environment Variables

Create a `.env.production` file with appropriate values:

```bash
# Application
APP_NAME=DSPy Meme Generator
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=your-secure-secret-key
ALLOWED_HOSTS=your-domain.com
CORS_ORIGINS=https://your-domain.com

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/dspy_meme_gen
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=1800
DATABASE_ECHO=false

# Cache
CACHE_URL=redis://redis:6379/0
CACHE_PREFIX=dspy_meme
CACHE_TTL=300
CACHE_MAX_CONNECTIONS=100

# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_CORS_ORIGINS=["https://your-domain.com"]
API_PREFIX=/api/v1
API_DOCS_URL=/docs

# DSPy
DSPY_MODEL_NAME=gpt-4
DSPY_TEMPERATURE=0.7
DSPY_MAX_TOKENS=1000
DSPY_RETRY_ATTEMPTS=3
DSPY_RETRY_DELAY=1

# Image Generation
IMAGE_PROVIDER=openai
IMAGE_API_KEY=your-api-key
IMAGE_DEFAULT_SIZE=1024x1024
IMAGE_DEFAULT_QUALITY=high
IMAGE_MAX_BATCH_SIZE=4

# External Services
OPENAI_API_KEY=your-openai-key
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# Performance
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000
KEEPALIVE=65
```

### SSL Certificate Setup

1. Generate SSL certificate:
```bash
certbot certonly --standalone -d yourdomain.com
```

2. Configure certificate paths in your reverse proxy.

## Docker Deployment

### Single Container

1. Build the image:
```bash
docker build -t dspy-meme-gen:latest .
```

2. Run the container:
```bash
docker run -d \
    --name dspy-meme-gen \
    -p 8000:8000 \
    --env-file .env.production \
    dspy-meme-gen:latest
```

### Docker Compose

1. Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    image: dspy-meme-gen:latest
    ports:
      - "8000:8000"
    env_file: .env.production
    depends_on:
      - db
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: dspy_meme_gen
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

2. Deploy with Docker Compose:
```bash
docker-compose -f docker-compose.yml up -d
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Helm 3

### Configuration

1. Create Kubernetes secrets:
```bash
kubectl create secret generic dspy-meme-gen-secrets \
    --from-file=.env.production
```

2. Create ConfigMap:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dspy-meme-gen-config
data:
  ENVIRONMENT: production
  WORKER_PROCESSES: "4"
  WORKER_CONNECTIONS: "1000"
```

### Deployment Files

1. Create deployment manifest (`k8s/deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-meme-gen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dspy-meme-gen
  template:
    metadata:
      labels:
        app: dspy-meme-gen
    spec:
      containers:
      - name: dspy-meme-gen
        image: dspy-meme-gen:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: dspy-meme-gen-secrets
        - configMapRef:
            name: dspy-meme-gen-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 15
```

2. Create service manifest (`k8s/service.yaml`):
```yaml
apiVersion: v1
kind: Service
metadata:
  name: dspy-meme-gen
spec:
  selector:
    app: dspy-meme-gen
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

3. Create ingress manifest (`k8s/ingress.yaml`):
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: dspy-meme-gen
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: dspy-meme-gen-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: dspy-meme-gen
            port:
              number: 80
```

### Deploy to Kubernetes

1. Apply manifests:
```bash
kubectl apply -f k8s/
```

2. Verify deployment:
```bash
kubectl get pods
kubectl get services
kubectl get ingress
```

## Cloud Platform Deployments

### AWS ECS

1. Create ECS cluster:
```bash
aws ecs create-cluster --cluster-name dspy-meme-gen
```

2. Create task definition:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

3. Create service:
```bash
aws ecs create-service \
  --cluster dspy-meme-gen \
  --service-name meme-gen \
  --task-definition dspy-meme-gen \
  --desired-count 3
```

### Google Cloud Run

1. Build and push image:
```bash
gcloud builds submit --tag gcr.io/your-project/dspy-meme-gen
```

2. Deploy service:
```bash
gcloud run deploy dspy-meme-gen \
  --image gcr.io/your-project/dspy-meme-gen \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Monitoring and Logging

### Prometheus Metrics

1. Add Prometheus annotations to service:
```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

2. Available metrics:
- `meme_generation_requests_total`
- `meme_generation_duration_seconds`
- `cache_hit_ratio`
- `database_connection_pool_size`

### Logging

1. Configure logging in production:
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        }
    },
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO'
    }
}
```

2. Log aggregation with Fluentd/ELK Stack.

## Backup and Recovery

### Database Backups

1. Automated daily backups:
```bash
kubectl create cronjob postgres-backup \
    --schedule="0 2 * * *" \
    --image=postgres:13-alpine \
    -- pg_dump -h db -U user dspy_meme_gen > backup.sql
```

2. Store backups in cloud storage:
```bash
gsutil cp backup.sql gs://your-bucket/backups/
```

### Recovery Procedures

1. Restore database:
```bash
psql -h db -U user dspy_meme_gen < backup.sql
```

2. Verify data integrity:
```bash
python manage.py check_data_integrity
```

## Performance Tuning

### Database Optimization

1. Configure connection pool:
```python
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 40
POOL_TIMEOUT = 30
POOL_RECYCLE = 1800
```

2. Add indexes:
```sql
CREATE INDEX idx_memes_created_at ON memes(created_at);
CREATE INDEX idx_templates_popularity ON templates(popularity_score);
```

### Caching Strategy

1. Configure Redis:
```python
CACHE_TTL = 300  # 5 minutes
CACHE_KEY_PREFIX = 'dspy_meme_gen:'
```

2. Cache patterns:
```python
# Cache templates
@cached(ttl=3600)
async def get_templates():
    return await db.fetch_templates()

# Cache meme results
@cached(ttl=300)
async def generate_meme(topic: str):
    return await pipeline.generate(topic)
```

## Security Considerations

1. Enable security headers:
```python
SECURE_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}
```

2. Configure CORS:
```python
CORS_ORIGINS = [
    'https://your-domain.com',
    'https://api.your-domain.com'
]
```

3. Rate limiting:
```python
RATE_LIMIT = {
    'DEFAULT': '100/hour',
    'BURST': '5/minute'
}
```

## Troubleshooting

### Common Issues

1. Database connection errors:
- Check connection string
- Verify network connectivity
- Check connection pool settings

2. Cache misses:
- Verify Redis connection
- Check TTL settings
- Monitor cache hit ratio

3. Performance issues:
- Check resource utilization
- Review database query plans
- Monitor external service latency

### Health Checks

1. Application health:
```bash
curl https://your-domain.com/health
```

2. Database health:
```bash
python manage.py check_db_health
```

3. Cache health:
```bash
python manage.py check_cache_health
```

### Debugging Steps

1. Check application logs:
```bash
kubectl logs deployment/dspy-meme-gen
```

2. Monitor resource usage:
```bash
kubectl top pods
```

3. Check service health:
```bash
kubectl describe service dspy-meme-gen
```

### Support Resources

- [Documentation](docs/)
- [GitHub Issues](https://github.com/yourusername/dspy-meme-gen/issues)
- [API Status Page](https://status.yourdomain.com) 