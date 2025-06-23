"""
Enhanced metrics collection with actor-specific metrics for the MemesPy system.
Provides comprehensive metrics for actors, pipelines, and system performance.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog
from opentelemetry import metrics
from opentelemetry.metrics import (
    Counter, Histogram, UpDownCounter, ObservableGauge,
    Observation, CallbackOptions
)

from .telemetry import get_meter, MetricNames

logger = structlog.get_logger(__name__)


class ActorMetrics:
    """Metrics collection for individual actors."""
    
    def __init__(self, actor_name: str):
        self.actor_name = actor_name
        self.meter = get_meter()
        self._lock = Lock()
        self._start_time = time.time()
        
        # Message processing metrics
        self._message_counter = self.meter.create_counter(
            name=f"actor.{actor_name}.messages.processed",
            description=f"Total messages processed by {actor_name}",
            unit="messages"
        )
        
        self._message_duration = self.meter.create_histogram(
            name=f"actor.{actor_name}.message.duration",
            description=f"Message processing duration for {actor_name}",
            unit="ms"
        )
        
        self._message_errors = self.meter.create_counter(
            name=f"actor.{actor_name}.messages.errors",
            description=f"Message processing errors for {actor_name}",
            unit="errors"
        )
        
        # Actor state metrics
        self._state_changes = self.meter.create_counter(
            name=f"actor.{actor_name}.state.changes",
            description=f"State changes for {actor_name}",
            unit="changes"
        )
        
        self._mailbox_size = self.meter.create_histogram(
            name=f"actor.{actor_name}.mailbox.size",
            description=f"Mailbox size for {actor_name}",
            unit="messages"
        )
        
        # Performance metrics
        self._cpu_usage = self.meter.create_histogram(
            name=f"actor.{actor_name}.cpu.usage",
            description=f"CPU usage for {actor_name}",
            unit="percent"
        )
        
        self._memory_usage = self.meter.create_histogram(
            name=f"actor.{actor_name}.memory.usage",
            description=f"Memory usage for {actor_name}",
            unit="MB"
        )
        
        # Internal tracking
        self._message_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._recent_durations: deque = deque(maxlen=100)
        self._uptime_start = time.time()
        
    def record_message_processed(
        self,
        message_type: str,
        duration_ms: float,
        success: bool = True,
        **attributes
    ):
        """Record metrics for a processed message."""
        with self._lock:
            # Update counters
            self._message_counts[message_type] += 1
            if not success:
                self._error_counts[message_type] += 1
            
            # Store recent duration
            self._recent_durations.append(duration_ms)
            
            # Record OpenTelemetry metrics
            metric_attributes = {
                "actor": self.actor_name,
                "message_type": message_type,
                "success": str(success),
                **attributes
            }
            
            self._message_counter.add(1, metric_attributes)
            self._message_duration.record(duration_ms, metric_attributes)
            
            if not success:
                self._message_errors.add(1, metric_attributes)
    
    def record_state_change(self, from_state: str, to_state: str, reason: str):
        """Record actor state change."""
        attributes = {
            "actor": self.actor_name,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason
        }
        
        self._state_changes.add(1, attributes)
        
        logger.info(
            "Actor state changed",
            actor=self.actor_name,
            from_state=from_state,
            to_state=to_state,
            reason=reason
        )
    
    def record_mailbox_size(self, size: int):
        """Record current mailbox size."""
        attributes = {"actor": self.actor_name}
        self._mailbox_size.record(size, attributes)
    
    def record_resource_usage(self, cpu_percent: float, memory_mb: float):
        """Record resource usage metrics."""
        attributes = {"actor": self.actor_name}
        
        self._cpu_usage.record(cpu_percent, attributes)
        self._memory_usage.record(memory_mb, attributes)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of actor metrics."""
        with self._lock:
            uptime = time.time() - self._uptime_start
            total_messages = sum(self._message_counts.values())
            total_errors = sum(self._error_counts.values())
            
            avg_duration = (
                sum(self._recent_durations) / len(self._recent_durations)
                if self._recent_durations else 0
            )
            
            return {
                "actor_name": self.actor_name,
                "uptime_seconds": uptime,
                "total_messages_processed": total_messages,
                "total_errors": total_errors,
                "error_rate": total_errors / total_messages if total_messages > 0 else 0,
                "average_processing_duration_ms": avg_duration,
                "messages_per_second": total_messages / uptime if uptime > 0 else 0,
                "message_counts_by_type": dict(self._message_counts),
                "error_counts_by_type": dict(self._error_counts)
            }


class PipelineMetrics:
    """Metrics collection for meme generation pipelines."""
    
    def __init__(self):
        self.meter = get_meter()
        self._lock = Lock()
        
        # Pipeline execution metrics
        self._pipeline_counter = self.meter.create_counter(
            name="pipeline.executions.total",
            description="Total pipeline executions",
            unit="executions"
        )
        
        self._pipeline_duration = self.meter.create_histogram(
            name="pipeline.execution.duration",
            description="Pipeline execution duration",
            unit="ms"
        )
        
        self._pipeline_success_rate = self.meter.create_counter(
            name="pipeline.executions.successful",
            description="Successful pipeline executions",
            unit="executions"
        )
        
        # Stage-specific metrics
        self._stage_duration = self.meter.create_histogram(
            name="pipeline.stage.duration",
            description="Pipeline stage duration",
            unit="ms"
        )
        
        self._stage_success_rate = self.meter.create_counter(
            name="pipeline.stage.successful",
            description="Successful pipeline stage executions",
            unit="executions"
        )
        
        # Quality metrics
        self._quality_scores = self.meter.create_histogram(
            name="pipeline.quality.scores",
            description="Meme quality score distribution",
            unit="score"
        )
        
        self._refinement_iterations = self.meter.create_histogram(
            name="pipeline.refinement.iterations",
            description="Number of refinement iterations",
            unit="iterations"
        )
        
        # Internal tracking
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
        self._completed_pipelines: deque = deque(maxlen=1000)
        
    def start_pipeline(
        self,
        pipeline_id: str,
        pipeline_type: str,
        topic: str,
        format_type: str
    ):
        """Start tracking a new pipeline execution."""
        with self._lock:
            self._active_pipelines[pipeline_id] = {
                "type": pipeline_type,
                "topic": topic,
                "format": format_type,
                "start_time": time.time(),
                "stages": {},
                "current_stage": None
            }
            
            # Record pipeline start
            attributes = {
                "pipeline_type": pipeline_type,
                "topic": topic,
                "format": format_type
            }
            self._pipeline_counter.add(1, attributes)
    
    def start_stage(self, pipeline_id: str, stage_name: str):
        """Start tracking a pipeline stage."""
        with self._lock:
            if pipeline_id in self._active_pipelines:
                pipeline = self._active_pipelines[pipeline_id]
                pipeline["current_stage"] = stage_name
                pipeline["stages"][stage_name] = {
                    "start_time": time.time(),
                    "completed": False
                }
    
    def complete_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        success: bool,
        **stage_data
    ):
        """Complete a pipeline stage."""
        with self._lock:
            if pipeline_id in self._active_pipelines:
                pipeline = self._active_pipelines[pipeline_id]
                
                if stage_name in pipeline["stages"]:
                    stage = pipeline["stages"][stage_name]
                    stage["completed"] = True
                    stage["success"] = success
                    stage["duration"] = time.time() - stage["start_time"]
                    stage.update(stage_data)
                    
                    # Record stage metrics
                    attributes = {
                        "pipeline_type": pipeline["type"],
                        "stage": stage_name,
                        "success": str(success),
                        "topic": pipeline["topic"],
                        "format": pipeline["format"]
                    }
                    
                    duration_ms = stage["duration"] * 1000
                    self._stage_duration.record(duration_ms, attributes)
                    
                    if success:
                        self._stage_success_rate.add(1, attributes)
    
    def complete_pipeline(
        self,
        pipeline_id: str,
        success: bool,
        quality_score: Optional[float] = None,
        refinement_count: int = 0,
        **pipeline_data
    ):
        """Complete a pipeline execution."""
        with self._lock:
            if pipeline_id not in self._active_pipelines:
                return
            
            pipeline = self._active_pipelines[pipeline_id]
            pipeline["end_time"] = time.time()
            pipeline["duration"] = pipeline["end_time"] - pipeline["start_time"]
            pipeline["success"] = success
            pipeline["quality_score"] = quality_score
            pipeline["refinement_count"] = refinement_count
            pipeline.update(pipeline_data)
            
            # Record pipeline completion metrics
            attributes = {
                "pipeline_type": pipeline["type"],
                "success": str(success),
                "topic": pipeline["topic"],
                "format": pipeline["format"]
            }
            
            duration_ms = pipeline["duration"] * 1000
            self._pipeline_duration.record(duration_ms, attributes)
            
            if success:
                self._pipeline_success_rate.add(1, attributes)
            
            # Record quality score
            if quality_score is not None:
                quality_attributes = {
                    "topic": pipeline["topic"],
                    "format": pipeline["format"]
                }
                self._quality_scores.record(quality_score, quality_attributes)
            
            # Record refinement iterations
            if refinement_count > 0:
                refinement_attributes = {
                    "topic": pipeline["topic"],
                    "format": pipeline["format"],
                    "success": str(success)
                }
                self._refinement_iterations.record(refinement_count, refinement_attributes)
            
            # Move to completed pipelines
            self._completed_pipelines.append(pipeline)
            del self._active_pipelines[pipeline_id]
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline metrics."""
        with self._lock:
            active_count = len(self._active_pipelines)
            completed_count = len(self._completed_pipelines)
            
            # Calculate success rate
            successful = sum(1 for p in self._completed_pipelines if p.get("success", False))
            success_rate = successful / completed_count if completed_count > 0 else 0
            
            # Calculate average duration
            avg_duration = (
                sum(p.get("duration", 0) for p in self._completed_pipelines) / completed_count
                if completed_count > 0 else 0
            )
            
            # Calculate average quality score
            quality_scores = [p.get("quality_score") for p in self._completed_pipelines if p.get("quality_score")]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                "active_pipelines": active_count,
                "completed_pipelines": completed_count,
                "success_rate": success_rate,
                "average_duration_seconds": avg_duration,
                "average_quality_score": avg_quality,
                "total_refinements": sum(p.get("refinement_count", 0) for p in self._completed_pipelines)
            }


class SystemMetrics:
    """System-wide metrics collection."""
    
    def __init__(self):
        self.meter = get_meter()
        self._lock = Lock()
        
        # System resource metrics
        self._system_cpu = self.meter.create_histogram(
            name="system.cpu.usage",
            description="System CPU usage",
            unit="percent"
        )
        
        self._system_memory = self.meter.create_histogram(
            name="system.memory.usage",
            description="System memory usage",
            unit="MB"
        )
        
        self._system_disk = self.meter.create_histogram(
            name="system.disk.usage",
            description="System disk usage",
            unit="percent"
        )
        
        # Connection pool metrics
        self._db_connections = self.meter.create_up_down_counter(
            name="system.database.connections.active",
            description="Active database connections",
            unit="connections"
        )
        
        self._cache_connections = self.meter.create_up_down_counter(
            name="system.cache.connections.active",
            description="Active cache connections",
            unit="connections"
        )
        
        # Request metrics
        self._api_requests = self.meter.create_counter(
            name="system.api.requests.total",
            description="Total API requests",
            unit="requests"
        )
        
        self._api_duration = self.meter.create_histogram(
            name="system.api.request.duration",
            description="API request duration",
            unit="ms"
        )
        
        # Error metrics
        self._system_errors = self.meter.create_counter(
            name="system.errors.total",
            description="Total system errors",
            unit="errors"
        )
        
        # Observable gauges for real-time metrics
        self.meter.create_observable_gauge(
            name="system.uptime.seconds",
            callbacks=[self._get_uptime],
            description="System uptime in seconds",
            unit="seconds"
        )
        
        # Internal tracking
        self._start_time = time.time()
        self._error_counts: Dict[str, int] = defaultdict(int)
        
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float
    ):
        """Record API request metrics."""
        attributes = {
            "method": method,
            "endpoint": self._normalize_endpoint(endpoint),
            "status_class": f"{status_code // 100}xx"
        }
        
        self._api_requests.add(1, attributes)
        self._api_duration.record(duration_ms, attributes)
    
    def record_system_resources(
        self,
        cpu_percent: float,
        memory_mb: float,
        disk_percent: float
    ):
        """Record system resource usage."""
        self._system_cpu.record(cpu_percent, {})
        self._system_memory.record(memory_mb, {})
        self._system_disk.record(disk_percent, {})
    
    def record_db_connection_change(self, delta: int):
        """Record database connection count change."""
        self._db_connections.add(delta, {"pool": "database"})
    
    def record_cache_connection_change(self, delta: int):
        """Record cache connection count change."""
        self._cache_connections.add(delta, {"pool": "cache"})
    
    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record system error."""
        with self._lock:
            self._error_counts[error_type] += 1
        
        attributes = {
            "error_type": error_type,
            "component": component,
            "severity": severity
        }
        
        self._system_errors.add(1, attributes)
    
    def _get_uptime(self, options: CallbackOptions):
        """Callback for uptime metric."""
        uptime = time.time() - self._start_time
        yield Observation(uptime, {})
    
    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize endpoint to reduce cardinality."""
        import re
        # Replace UUIDs and numeric IDs
        endpoint = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}', endpoint
        )
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        return endpoint
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        with self._lock:
            uptime = time.time() - self._start_time
            total_errors = sum(self._error_counts.values())
            
            return {
                "uptime_seconds": uptime,
                "total_errors": total_errors,
                "error_counts_by_type": dict(self._error_counts),
                "uptime_hours": uptime / 3600
            }


class MetricsAggregator:
    """Aggregates and manages all metrics collectors."""
    
    def __init__(self):
        self.actor_metrics: Dict[str, ActorMetrics] = {}
        self.pipeline_metrics = PipelineMetrics()
        self.system_metrics = SystemMetrics()
        self._lock = Lock()
        
        # Background task for periodic metrics collection
        self._collection_task: Optional[asyncio.Task] = None
        self._collection_interval = 30  # seconds
        
    def get_actor_metrics(self, actor_name: str) -> ActorMetrics:
        """Get or create metrics collector for an actor."""
        with self._lock:
            if actor_name not in self.actor_metrics:
                self.actor_metrics[actor_name] = ActorMetrics(actor_name)
            return self.actor_metrics[actor_name]
    
    def remove_actor_metrics(self, actor_name: str):
        """Remove metrics collector for a terminated actor."""
        with self._lock:
            if actor_name in self.actor_metrics:
                del self.actor_metrics[actor_name]
    
    def start_background_collection(self):
        """Start background task for periodic metrics collection."""
        if self._collection_task is None or self._collection_task.done():
            self._collection_task = asyncio.create_task(self._collect_system_metrics())
    
    def stop_background_collection(self):
        """Stop background metrics collection."""
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        import psutil
        
        while True:
            try:
                # Collect system resource metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.system_metrics.record_system_resources(
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    disk_percent=disk.percent
                )
                
                # Collect per-actor resource metrics if possible
                for actor_name, actor_metrics in self.actor_metrics.items():
                    try:
                        # This would need to be implemented per actor if we track processes
                        actor_metrics.record_resource_usage(
                            cpu_percent=cpu_percent / len(self.actor_metrics),
                            memory_mb=memory.used / 1024 / 1024 / len(self.actor_metrics)
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to collect actor metrics",
                            actor=actor_name,
                            error=str(e)
                        )
                
                await asyncio.sleep(self._collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection", error=str(e))
                await asyncio.sleep(self._collection_interval)
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.system_metrics.get_system_summary(),
            "pipelines": self.pipeline_metrics.get_pipeline_summary(),
            "actors": {}
        }
        
        # Add actor summaries
        with self._lock:
            for actor_name, actor_metrics in self.actor_metrics.items():
                summary["actors"][actor_name] = actor_metrics.get_summary()
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        # This would be handled by the OpenTelemetry Prometheus exporter
        # This method is a placeholder for custom export logic
        summary = self.get_comprehensive_summary()
        
        lines = []
        
        # System metrics
        system = summary["system"]
        lines.append(f'# HELP system_uptime_seconds System uptime')
        lines.append(f'# TYPE system_uptime_seconds gauge')
        lines.append(f'system_uptime_seconds {system["uptime_seconds"]}')
        
        lines.append(f'# HELP system_errors_total Total system errors')
        lines.append(f'# TYPE system_errors_total counter')
        lines.append(f'system_errors_total {system["total_errors"]}')
        
        # Actor metrics
        for actor_name, actor_data in summary["actors"].items():
            lines.append(f'# HELP actor_messages_total Total messages processed by actor')
            lines.append(f'# TYPE actor_messages_total counter')
            lines.append(f'actor_messages_total{{actor="{actor_name}"}} {actor_data["total_messages_processed"]}')
            
            lines.append(f'# HELP actor_errors_total Total errors by actor')
            lines.append(f'# TYPE actor_errors_total counter')
            lines.append(f'actor_errors_total{{actor="{actor_name}"}} {actor_data["total_errors"]}')
        
        # Pipeline metrics
        pipelines = summary["pipelines"]
        lines.append(f'# HELP pipeline_executions_total Total pipeline executions')
        lines.append(f'# TYPE pipeline_executions_total counter')
        lines.append(f'pipeline_executions_total {pipelines["completed_pipelines"]}')
        
        lines.append(f'# HELP pipeline_success_rate Pipeline success rate')
        lines.append(f'# TYPE pipeline_success_rate gauge')
        lines.append(f'pipeline_success_rate {pipelines["success_rate"]}')
        
        return '\n'.join(lines)


# Global metrics aggregator
metrics_aggregator = MetricsAggregator()

# Export public API
__all__ = [
    "ActorMetrics",
    "PipelineMetrics", 
    "SystemMetrics",
    "MetricsAggregator",
    "metrics_aggregator"
]