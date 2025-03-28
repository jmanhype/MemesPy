"""Tests for monitoring and metrics functionality."""

from typing import Any, Dict, List, Optional, cast
import pytest
import asyncio
import time
from prometheus_client import REGISTRY
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from dspy_meme_gen.monitoring.metrics import (
    MetricsMiddleware,
    MetricsCollector,
    setup_metrics,
    GENERATION_TIME,
    GENERATION_TOTAL,
    AGENT_EXECUTION_TIME,
    AGENT_SUCCESS_RATE,
    AGENT_FAILURES,
    EXTERNAL_SERVICE_LATENCY,
    EXTERNAL_SERVICE_ERRORS,
    CACHE_OPERATIONS,
    CACHE_LATENCY,
    DB_OPERATION_LATENCY
)
from dspy_meme_gen.monitoring.decorators import (
    track_agent_metrics,
    track_external_service,
    track_cache_operation,
    track_db_operation,
    track_meme_generation
)

@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application.
    
    Returns:
        A FastAPI application instance.
    """
    app = FastAPI()
    setup_metrics(app)
    return app

@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client.
    
    Args:
        app: The FastAPI application
        
    Returns:
        A TestClient instance.
    """
    return TestClient(app)

@pytest.fixture
def reset_registry() -> None:
    """Reset the Prometheus registry between tests."""
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)

class TestMetricsMiddleware:
    """Tests for the MetricsMiddleware class."""
    
    async def test_request_tracking(self, app: FastAPI, client: TestClient, reset_registry: None) -> None:
        """Test that request metrics are tracked correctly."""
        @app.get("/test")
        async def test_endpoint() -> Dict[str, str]:
            return {"status": "ok"}
            
        response = client.get("/test")
        assert response.status_code == 200
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "http_request_duration_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if sample.labels["path"] == "/test":
                        assert sample.labels["method"] == "GET"
                        assert sample.labels["status"] == "200"
                        assert sample.value > 0
                        
    async def test_error_tracking(self, app: FastAPI, client: TestClient, reset_registry: None) -> None:
        """Test that error metrics are tracked correctly."""
        @app.get("/error")
        async def error_endpoint() -> None:
            raise ValueError("Test error")
            
        response = client.get("/error")
        assert response.status_code == 500
        
        # Check error metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "http_request_duration_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if sample.labels["path"] == "/error":
                        assert sample.labels["method"] == "GET"
                        assert sample.labels["status"] == "500"
                        assert sample.value > 0

class TestMetricsCollector:
    """Tests for the MetricsCollector class."""
    
    async def test_track_meme_generation(self, reset_registry: None) -> None:
        """Test tracking meme generation metrics."""
        start_time = time.time()
        await MetricsCollector.track_meme_generation(
            template_type="test_template",
            start_time=start_time,
            status="success"
        )
        
        # Check generation metrics
        for metric in REGISTRY.collect():
            if metric.name == "meme_generation_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if sample.labels["template_type"] == "test_template":
                        assert sample.labels["status"] == "success"
                        assert sample.value > 0
                        
    async def test_track_agent_execution(self, reset_registry: None) -> None:
        """Test tracking agent execution metrics."""
        start_time = time.time()
        await MetricsCollector.track_agent_execution(
            agent_name="test_agent",
            start_time=start_time,
            success=True
        )
        
        # Check agent metrics
        for metric in REGISTRY.collect():
            if metric.name == "agent_execution_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if sample.labels["agent_name"] == "test_agent":
                        assert sample.value > 0

class TestDecorators:
    """Tests for metric decorators."""
    
    async def test_track_agent_metrics_decorator(self, reset_registry: None) -> None:
        """Test the track_agent_metrics decorator."""
        @track_agent_metrics("test_agent")
        async def test_function() -> str:
            return "success"
            
        result = await test_function()
        assert result == "success"
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "agent_execution_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if sample.labels["agent_name"] == "test_agent":
                        assert sample.value > 0
                        
    async def test_track_external_service_decorator(self, reset_registry: None) -> None:
        """Test the track_external_service decorator."""
        @track_external_service("test_service", "test_operation")
        async def test_function() -> str:
            return "success"
            
        result = await test_function()
        assert result == "success"
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "external_service_latency_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["service_name"] == "test_service" and 
                        sample.labels["operation"] == "test_operation"):
                        assert sample.value > 0
                        
    async def test_track_cache_operation_decorator(self, reset_registry: None) -> None:
        """Test the track_cache_operation decorator."""
        @track_cache_operation("test_cache")
        async def test_function() -> Optional[str]:
            return "cached_value"
            
        result = await test_function()
        assert result == "cached_value"
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "cache_operations_total":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["cache_type"] == "test_cache" and 
                        sample.labels["operation"] == "hit"):
                        assert sample.value > 0
                        
    async def test_track_db_operation_decorator(self, reset_registry: None) -> None:
        """Test the track_db_operation decorator."""
        @track_db_operation("select", "test_table")
        async def test_function() -> List[Dict[str, Any]]:
            return [{"id": 1, "name": "test"}]
            
        result = await test_function()
        assert len(result) == 1
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "db_operation_latency_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["operation"] == "select" and 
                        sample.labels["table"] == "test_table"):
                        assert sample.value > 0
                        
    async def test_track_meme_generation_decorator(self, reset_registry: None) -> None:
        """Test the track_meme_generation decorator."""
        @track_meme_generation("test_template")
        async def test_function() -> Dict[str, Any]:
            return {"status": "success"}
            
        result = await test_function()
        assert result["status"] == "success"
        
        # Check metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "meme_generation_seconds":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["template_type"] == "test_template" and 
                        sample.labels["status"] == "success"):
                        assert sample.value > 0

class TestErrorHandling:
    """Tests for error handling in metrics collection."""
    
    async def test_agent_error_tracking(self, reset_registry: None) -> None:
        """Test that agent errors are tracked correctly."""
        @track_agent_metrics("test_agent")
        async def failing_function() -> None:
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            await failing_function()
            
        # Check error metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "agent_failures_total":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["agent_name"] == "test_agent" and 
                        sample.labels["error_type"] == "ValueError"):
                        assert sample.value > 0
                        
    async def test_external_service_error_tracking(self, reset_registry: None) -> None:
        """Test that external service errors are tracked correctly."""
        @track_external_service("test_service", "test_operation")
        async def failing_function() -> None:
            raise ConnectionError("Test error")
            
        with pytest.raises(ConnectionError):
            await failing_function()
            
        # Check error metrics were recorded
        for metric in REGISTRY.collect():
            if metric.name == "external_service_errors_total":
                assert len(metric.samples) > 0
                for sample in metric.samples:
                    if (sample.labels["service_name"] == "test_service" and 
                        sample.labels["error_type"] == "ConnectionError"):
                        assert sample.value > 0 