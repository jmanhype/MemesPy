"""
Metric collection design with cardinality control strategies.
Implements best practices for preventing metric explosion while maintaining observability.
"""

from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
from collections import defaultdict
import time

from opentelemetry.metrics import (
    Meter, Counter, Histogram, UpDownCounter, 
    ObservableGauge, CallbackOptions, Observation
)
from structlog import get_logger

from .telemetry import get_meter

logger = get_logger(__name__)

class CardinalityControl:
    """Strategies for controlling metric cardinality."""
    
    # Maximum allowed cardinality per metric
    MAX_CARDINALITY = 10000
    
    # Maximum unique values per label
    MAX_LABEL_VALUES = {
        "endpoint": 100,
        "agent_name": 50,
        "format_type": 20,
        "error_type": 50,
        "cache_key": 1000,
        "user_id": 5000
    }
    
    @staticmethod
    def normalize_endpoint(endpoint: str) -> str:
        """Normalize API endpoints to reduce cardinality."""
        # Replace UUIDs
        endpoint = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{uuid}',
            endpoint
        )
        
        # Replace numeric IDs
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        
        # Replace query parameters
        endpoint = re.sub(r'\?.*$', '', endpoint)
        
        # Normalize common patterns
        patterns = [
            (r'/v\d+/', '/v{version}/'),
            (r'/users/[^/]+/', '/users/{username}/'),
            (r'/items/[^/]+/', '/items/{item}/'),
        ]
        
        for pattern, replacement in patterns:
            endpoint = re.sub(pattern, replacement, endpoint)
        
        return endpoint
    
    @staticmethod
    def normalize_error_type(error: Exception) -> str:
        """Normalize error types to reduce cardinality."""
        error_class = type(error).__name__
        
        # Group similar errors
        error_groups = {
            "ValidationError": ["ValidationError", "ValueError", "TypeError"],
            "NetworkError": ["ConnectionError", "TimeoutError", "HTTPError"],
            "DatabaseError": ["IntegrityError", "OperationalError", "DataError"],
            "AuthError": ["AuthenticationError", "PermissionError", "Unauthorized"],
        }
        
        for group, errors in error_groups.items():
            if error_class in errors:
                return group
        
        # Default to generic error class
        return error_class if error_class in ["ValueError", "TypeError", "KeyError"] else "OtherError"
    
    @staticmethod
    def hash_high_cardinality_value(value: str, prefix: str = "") -> str:
        """Hash high cardinality values to fixed set."""
        # Use first 8 chars of hash to limit cardinality
        hash_value = hashlib.md5(value.encode()).hexdigest()[:8]
        return f"{prefix}{hash_value}" if prefix else hash_value
    
    @staticmethod
    def bucket_numeric_value(value: float, buckets: List[float]) -> str:
        """Bucket numeric values to reduce cardinality."""
        for i, bucket in enumerate(buckets):
            if value <= bucket:
                if i == 0:
                    return f"<={bucket}"
                else:
                    return f"{buckets[i-1]}-{bucket}"
        return f">{buckets[-1]}"

@dataclass
class MetricDefinition:
    """Definition of a metric with cardinality controls."""
    name: str
    type: str  # counter, histogram, gauge
    unit: str
    description: str
    allowed_labels: Set[str]
    label_normalizers: Dict[str, Callable[[Any], str]]
    cardinality_limit: int = 1000

class CardinalityLimiter:
    """Enforces cardinality limits on metrics."""
    
    def __init__(self, max_cardinality: int = 10000):
        self.max_cardinality = max_cardinality
        self.label_combinations: Dict[str, Set[tuple]] = defaultdict(set)
        self.dropped_labels: Dict[str, int] = defaultdict(int)
    
    def check_and_add(self, metric_name: str, labels: Dict[str, str]) -> bool:
        """Check if label combination is within cardinality limit."""
        label_tuple = tuple(sorted(labels.items()))
        
        if label_tuple in self.label_combinations[metric_name]:
            return True
        
        if len(self.label_combinations[metric_name]) >= self.max_cardinality:
            self.dropped_labels[metric_name] += 1
            return False
        
        self.label_combinations[metric_name].add(label_tuple)
        return True
    
    def get_cardinality_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cardinality statistics for monitoring."""
        return {
            metric: {
                "cardinality": len(combinations),
                "dropped": self.dropped_labels[metric]
            }
            for metric, combinations in self.label_combinations.items()
        }

class MetricsDesign:
    """Comprehensive metrics design with cardinality control."""
    
    def __init__(self):
        self.meter = get_meter()
        self.cardinality_limiter = CardinalityLimiter()
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all metrics with proper cardinality controls."""
        
        # Request metrics (low cardinality)
        self.request_duration = self.meter.create_histogram(
            name="meme.request.duration",
            unit="ms",
            description="HTTP request duration"
        )
        
        self.request_count = self.meter.create_counter(
            name="meme.request.total",
            unit="1",
            description="Total HTTP requests"
        )
        
        # Pipeline metrics (medium cardinality)
        self.pipeline_stage_duration = self.meter.create_histogram(
            name="meme.pipeline.stage.duration",
            unit="ms",
            description="Pipeline stage execution time"
        )
        
        self.pipeline_success_rate = self.meter.create_counter(
            name="meme.pipeline.success",
            unit="1",
            description="Successful pipeline executions"
        )
        
        # Agent metrics (controlled cardinality)
        self.agent_execution_time = self.meter.create_histogram(
            name="meme.agent.execution.time",
            unit="ms",
            description="Agent execution time"
        )
        
        self.agent_error_count = self.meter.create_counter(
            name="meme.agent.errors",
            unit="1",
            description="Agent execution errors"
        )
        
        # Quality metrics (low cardinality through bucketing)
        self.quality_score_distribution = self.meter.create_histogram(
            name="meme.quality.score",
            unit="1",
            description="Meme quality score distribution"
        )
        
        # Cache metrics (controlled through hashing)
        self.cache_operations = self.meter.create_counter(
            name="meme.cache.operations",
            unit="1",
            description="Cache operations"
        )
        
        # Resource metrics (callback-based, no cardinality issues)
        self.meter.create_observable_gauge(
            name="meme.resource.memory",
            callbacks=[self._observe_memory],
            unit="MB",
            description="Memory usage"
        )
        
        self.meter.create_observable_gauge(
            name="meme.resource.connections",
            callbacks=[self._observe_connections],
            unit="1",
            description="Active connections"
        )
    
    def record_request(
        self,
        duration_ms: float,
        method: str,
        endpoint: str,
        status_code: int,
        user_id: Optional[str] = None
    ):
        """Record HTTP request with cardinality control."""
        # Normalize labels
        normalized_endpoint = CardinalityControl.normalize_endpoint(endpoint)
        status_class = f"{status_code // 100}xx"
        
        # Hash user_id if present
        user_bucket = None
        if user_id:
            user_bucket = CardinalityControl.hash_high_cardinality_value(
                user_id, prefix="user_"
            )[:4]  # Further limit cardinality
        
        labels = {
            "method": method,
            "endpoint": normalized_endpoint,
            "status": status_class
        }
        
        if user_bucket:
            labels["user_bucket"] = user_bucket
        
        # Check cardinality
        if self.cardinality_limiter.check_and_add("request", labels):
            self.request_duration.record(duration_ms, labels)
            self.request_count.add(1, labels)
    
    def record_pipeline_stage(
        self,
        stage: str,
        duration_ms: float,
        success: bool,
        topic_category: Optional[str] = None
    ):
        """Record pipeline stage execution with controlled labels."""
        # Limit stage names to predefined set
        allowed_stages = {
            "router", "trend_scan", "format_select", 
            "generation", "verification", "refinement"
        }
        
        if stage not in allowed_stages:
            stage = "other"
        
        # Categorize topics to reduce cardinality
        if topic_category:
            topic_categories = {
                "tech": ["ai", "programming", "crypto", "tech"],
                "entertainment": ["movies", "games", "music", "tv"],
                "news": ["politics", "world", "breaking", "current"],
                "lifestyle": ["food", "travel", "fitness", "fashion"],
                "other": []
            }
            
            categorized = False
            for category, keywords in topic_categories.items():
                if any(keyword in topic_category.lower() for keyword in keywords):
                    topic_category = category
                    categorized = True
                    break
            
            if not categorized:
                topic_category = "other"
        
        labels = {
            "stage": stage,
            "success": str(success).lower()
        }
        
        if topic_category:
            labels["topic_category"] = topic_category
        
        if self.cardinality_limiter.check_and_add("pipeline_stage", labels):
            self.pipeline_stage_duration.record(duration_ms, labels)
            if success:
                self.pipeline_success_rate.add(1, labels)
    
    def record_agent_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Record agent execution with normalized labels."""
        # Limit agent names
        allowed_agents = {
            "router", "trend_scanner", "format_selector",
            "prompt_generator", "image_renderer", "factuality",
            "appropriateness", "scorer", "refinement"
        }
        
        if agent_name not in allowed_agents:
            agent_name = "other"
        
        labels = {
            "agent": agent_name,
            "success": str(success).lower()
        }
        
        if self.cardinality_limiter.check_and_add("agent_execution", labels):
            self.agent_execution_time.record(duration_ms, labels)
        
        if not success and error_type:
            error_labels = {
                "agent": agent_name,
                "error_type": CardinalityControl.normalize_error_type(error_type)
            }
            
            if self.cardinality_limiter.check_and_add("agent_errors", error_labels):
                self.agent_error_count.add(1, error_labels)
    
    def record_quality_score(
        self,
        score: float,
        format_type: str,
        refinement_count: int = 0
    ):
        """Record quality score with bucketing."""
        # Bucket the score
        score_bucket = CardinalityControl.bucket_numeric_value(
            score,
            buckets=[0.2, 0.4, 0.6, 0.8, 1.0]
        )
        
        # Limit format types
        allowed_formats = {
            "standard", "drake", "distracted", "comparison",
            "expanding_brain", "custom"
        }
        
        if format_type not in allowed_formats:
            format_type = "other"
        
        # Bucket refinement count
        refinement_bucket = CardinalityControl.bucket_numeric_value(
            refinement_count,
            buckets=[0, 1, 3, 5, 10]
        )
        
        labels = {
            "score_bucket": score_bucket,
            "format": format_type,
            "refinements": refinement_bucket
        }
        
        if self.cardinality_limiter.check_and_add("quality_score", labels):
            self.quality_score_distribution.record(score, labels)
    
    def record_cache_operation(
        self,
        operation: str,
        cache_type: str,
        key: str,
        hit: bool,
        duration_ms: float
    ):
        """Record cache operation with key hashing."""
        # Hash the cache key to control cardinality
        key_hash = CardinalityControl.hash_high_cardinality_value(key)[:6]
        
        # Limit operation types
        allowed_operations = {"get", "set", "delete", "expire"}
        if operation not in allowed_operations:
            operation = "other"
        
        # Limit cache types
        allowed_cache_types = {"redis", "memory", "disk"}
        if cache_type not in allowed_cache_types:
            cache_type = "other"
        
        labels = {
            "operation": operation,
            "cache_type": cache_type,
            "hit": str(hit).lower(),
            "key_prefix": key_hash[:3]  # Further reduce cardinality
        }
        
        if self.cardinality_limiter.check_and_add("cache_operations", labels):
            self.cache_operations.add(1, labels)
    
    def _observe_memory(self, options: CallbackOptions) -> Iterable[Observation]:
        """Observe memory usage (no cardinality issues)."""
        import psutil
        memory = psutil.virtual_memory()
        yield Observation(
            memory.used / 1024 / 1024,
            {"type": "used"}
        )
        yield Observation(
            memory.available / 1024 / 1024,
            {"type": "available"}
        )
    
    def _observe_connections(self, options: CallbackOptions) -> Iterable[Observation]:
        """Observe connection counts (low cardinality)."""
        # This would connect to your connection pools
        # For now, returning dummy data
        yield Observation(10, {"pool": "database"})
        yield Observation(5, {"pool": "redis"})
    
    def get_cardinality_report(self) -> Dict[str, Any]:
        """Get cardinality statistics for monitoring."""
        stats = self.cardinality_limiter.get_cardinality_stats()
        
        return {
            "timestamp": time.time(),
            "metrics": stats,
            "total_combinations": sum(
                stat["cardinality"] for stat in stats.values()
            ),
            "total_dropped": sum(
                stat["dropped"] for stat in stats.values()
            )
        }

# Best practices for metric design
class MetricBestPractices:
    """
    Best practices for metric design to prevent cardinality explosion:
    
    1. **Use Static Labels**: Prefer labels with known, limited values
    2. **Normalize Dynamic Values**: Convert IDs to patterns, hash high-cardinality values
    3. **Bucket Numeric Values**: Use histograms or bucket continuous values
    4. **Limit Label Combinations**: Be careful with label multiplication
    5. **Use Exemplars**: Store high-cardinality data as exemplars, not labels
    6. **Monitor Cardinality**: Track and alert on cardinality growth
    7. **Use Aggregation**: Pre-aggregate where possible
    8. **Separate Metrics**: Use different metrics for different cardinality levels
    """
    
    @staticmethod
    def validate_metric_design(
        metric_name: str,
        labels: Dict[str, Any],
        max_cardinality: int = 10000
    ) -> tuple[bool, str]:
        """Validate metric design for cardinality issues."""
        # Check for high-cardinality labels
        high_cardinality_labels = []
        
        for label, value in labels.items():
            if label in ["user_id", "session_id", "request_id", "trace_id"]:
                high_cardinality_labels.append(label)
            elif isinstance(value, str) and len(set(value)) > 50:
                high_cardinality_labels.append(label)
        
        if high_cardinality_labels:
            return False, f"High cardinality labels detected: {high_cardinality_labels}"
        
        # Estimate cardinality
        estimated_cardinality = 1
        for label, value in labels.items():
            if isinstance(value, list):
                estimated_cardinality *= len(value)
            elif isinstance(value, str):
                # Rough estimate based on common patterns
                if "id" in label.lower():
                    estimated_cardinality *= 1000
                else:
                    estimated_cardinality *= 10
        
        if estimated_cardinality > max_cardinality:
            return False, f"Estimated cardinality ({estimated_cardinality}) exceeds limit"
        
        return True, "Metric design is valid"

# Global metrics instance
metrics_design = MetricsDesign()

__all__ = [
    "CardinalityControl",
    "MetricDefinition",
    "CardinalityLimiter",
    "MetricsDesign",
    "MetricBestPractices",
    "metrics_design"
]