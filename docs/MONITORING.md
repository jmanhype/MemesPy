# Monitoring and Observability

## Overview

The DSPy Meme Generation pipeline implements comprehensive monitoring and observability using Prometheus for metrics collection, Grafana for visualization, and structured logging for error tracking and analytics.

## Prometheus Metrics

### Core Metrics

1. **Generation Pipeline Metrics**
```python
# Time spent in meme generation
GENERATION_TIME = Histogram(
    "meme_generation_seconds",
    "Time spent generating memes",
    ["status", "template_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Generation success/failure counters
GENERATION_TOTAL = Counter(
    "meme_generation_total",
    "Total number of meme generation attempts",
    ["status"]
)

# Quality scores
MEME_QUALITY_SCORE = Histogram(
    "meme_quality_score",
    "Distribution of meme quality scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
```

2. **Agent Performance Metrics**
```python
# Agent execution time
AGENT_EXECUTION_TIME = Histogram(
    "agent_execution_seconds",
    "Time spent in each agent",
    ["agent_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Agent success rate
AGENT_SUCCESS_RATE = Counter(
    "agent_success_total",
    "Successful agent operations",
    ["agent_name"]
)

# Agent failures
AGENT_FAILURES = Counter(
    "agent_failures_total",
    "Failed agent operations",
    ["agent_name", "error_type"]
)
```

3. **External Service Metrics**
```python
# External API latency
EXTERNAL_SERVICE_LATENCY = Histogram(
    "external_service_latency_seconds",
    "External service request latency",
    ["service_name", "operation"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# External API errors
EXTERNAL_SERVICE_ERRORS = Counter(
    "external_service_errors_total",
    "External service errors",
    ["service_name", "error_type"]
)
```

4. **Cache Performance**
```python
# Cache operations
CACHE_OPERATIONS = Counter(
    "cache_operations_total",
    "Cache operations (hits/misses)",
    ["operation", "cache_type"]
)

# Cache latency
CACHE_LATENCY = Histogram(
    "cache_latency_seconds",
    "Cache operation latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)
```

5. **Database Metrics**
```python
# Database operation latency
DB_OPERATION_LATENCY = Histogram(
    "db_operation_latency_seconds",
    "Database operation latency",
    ["operation", "table"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Connection pool metrics
DB_POOL_SIZE = Gauge(
    "db_pool_size",
    "Current database connection pool size",
)

DB_POOL_AVAILABLE = Gauge(
    "db_pool_available_connections",
    "Available connections in the pool",
)
```

### Metric Collection

Configure Prometheus scraping in your application:

```python
from prometheus_client import start_http_server, REGISTRY
from prometheus_client.exposition import generate_latest

class MetricsMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.observe(duration)
            REQUEST_COUNT.inc()
            
        return response

# Start metrics server
start_http_server(8000)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dspy-meme-gen'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

## Grafana Dashboards

### 1. Overview Dashboard

```json
{
  "title": "DSPy Meme Generator Overview",
  "panels": [
    {
      "title": "Meme Generation Rate",
      "type": "graph",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "rate(meme_generation_total[5m])",
          "legendFormat": "{{status}}"
        }
      ]
    },
    {
      "title": "Generation Latency",
      "type": "heatmap",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "rate(meme_generation_seconds_bucket[5m])"
        }
      ]
    }
  ]
}
```

### 2. Performance Dashboard

```json
{
  "title": "Performance Metrics",
  "panels": [
    {
      "title": "Agent Performance",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(agent_execution_seconds_sum[5m]) / rate(agent_execution_seconds_count[5m])",
          "legendFormat": "{{agent_name}}"
        }
      ]
    },
    {
      "title": "External Service Latency",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(external_service_latency_seconds_sum[5m]) / rate(external_service_latency_seconds_count[5m])",
          "legendFormat": "{{service_name}}"
        }
      ]
    }
  ]
}
```

### 3. Error Tracking Dashboard

```json
{
  "title": "Error Tracking",
  "panels": [
    {
      "title": "Error Rate by Type",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(agent_failures_total[5m])",
          "legendFormat": "{{error_type}}"
        }
      ]
    },
    {
      "title": "External Service Errors",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(external_service_errors_total[5m])",
          "legendFormat": "{{service_name}} - {{error_type}}"
        }
      ]
    }
  ]
}
```

## Error Tracking

### Structured Logging

Configure structured logging with correlation IDs:

```python
import structlog
from typing import Optional

logger = structlog.get_logger()

class LoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        
        log = logger.bind(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method
        )
        
        try:
            response = await call_next(request)
            log.info("request_processed", status_code=response.status_code)
            return response
        except Exception as e:
            log.error("request_failed", 
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc()
            )
            raise
```

### Error Aggregation

Integration with error tracking services:

```python
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    environment=settings.ENVIRONMENT,
    traces_sample_rate=0.1,
    profiles_sample_rate=0.1,
)

app.add_middleware(SentryAsgiMiddleware)
```

## Usage Analytics

### Event Tracking

```python
from typing import Dict, Any

class AnalyticsEvent:
    def __init__(self, event_type: str, properties: Dict[str, Any]):
        self.event_type = event_type
        self.properties = properties
        self.timestamp = datetime.utcnow()

class AnalyticsTracker:
    async def track_event(self, event: AnalyticsEvent):
        # Store event in ClickHouse for analytics
        await clickhouse.insert([{
            "event_type": event.event_type,
            "properties": json.dumps(event.properties),
            "timestamp": event.timestamp
        }])
```

### Key Events

1. Meme Generation Events:
```python
await analytics.track_event(AnalyticsEvent(
    "meme_generated",
    {
        "template_id": template.id,
        "generation_time": generation_time,
        "quality_score": quality_score,
        "user_id": user_id
    }
))
```

2. User Interaction Events:
```python
await analytics.track_event(AnalyticsEvent(
    "meme_interaction",
    {
        "meme_id": meme_id,
        "interaction_type": "share",
        "platform": "twitter",
        "user_id": user_id
    }
))
```

### Analytics Queries

```sql
-- Popular templates
SELECT 
    template_id,
    COUNT(*) as usage_count,
    AVG(properties.quality_score) as avg_quality
FROM analytics_events
WHERE event_type = 'meme_generated'
GROUP BY template_id
ORDER BY usage_count DESC;

-- User engagement
SELECT 
    DATE_TRUNC('day', timestamp) as date,
    COUNT(DISTINCT properties.user_id) as daily_active_users,
    COUNT(*) as total_generations
FROM analytics_events
WHERE event_type = 'meme_generated'
GROUP BY date
ORDER BY date;
```

## Alerting Rules

### Prometheus Alerting

```yaml
groups:
  - name: dspy-meme-gen
    rules:
      - alert: HighErrorRate
        expr: rate(meme_generation_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in meme generation"
          
      - alert: SlowGeneration
        expr: histogram_quantile(0.95, rate(meme_generation_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow meme generation detected"
          
      - alert: ExternalServiceErrors
        expr: rate(external_service_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of external service errors"
```

### Alert Notifications

Configure alert notifications in Grafana:

```yaml
apiVersion: 1
notifications:
  - name: DevOps Team
    type: slack
    settings:
      url: https://hooks.slack.com/services/your-webhook-url
    
  - name: Emergency
    type: pagerduty
    settings:
      integrationKey: your-pagerduty-key
```

## Performance Monitoring

### Resource Usage Tracking

```python
RESOURCE_USAGE = Gauge(
    "resource_usage",
    "System resource usage",
    ["resource_type"]
)

async def track_resource_usage():
    while True:
        RESOURCE_USAGE.labels("cpu").set(psutil.cpu_percent())
        RESOURCE_USAGE.labels("memory").set(psutil.virtual_memory().percent)
        RESOURCE_USAGE.labels("disk").set(psutil.disk_usage("/").percent)
        await asyncio.sleep(60)
```

### Performance Profiling

```python
from pyinstrument import Profiler

class ProfilingMiddleware:
    async def __call__(self, request: Request, call_next):
        if request.query_params.get("profile"):
            profiler = Profiler()
            profiler.start()
            
            try:
                response = await call_next(request)
            finally:
                profiler.stop()
                
                if response.status_code == 200:
                    return HTMLResponse(profiler.output_html())
                
            return response
        return await call_next(request)
``` 