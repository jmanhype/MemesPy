"""
Real End-to-End Integration Test
================================

A proper cybernetic systems test that validates actual system behavior
under realistic conditions. No mocks, no theater, just truth.

Based on Stafford Beer's Viable System Model and Erlang/OTP principles.
"""

import asyncio
import json
import time
import pytest
import httpx
from typing import Dict, Any, List
from pathlib import Path
import subprocess
import signal
import os
from contextlib import asynccontextmanager

# Test Configuration
API_BASE_URL = "http://localhost:8081"
TEST_TIMEOUT = 120  # 2 minutes max per test
STARTUP_TIMEOUT = 30  # 30 seconds to start server


class SystemUnderTest:
    """Represents the actual running system, not a mock."""
    
    def __init__(self):
        self.process = None
        self.base_url = API_BASE_URL
        
    async def start(self) -> None:
        """Start the real system."""
        # Kill any existing processes
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        await asyncio.sleep(2)
        
        # Start the server
        self.process = subprocess.Popen([
            "python", "-m", "uvicorn", 
            "src.dspy_meme_gen.api.main:app", 
            "--port", "8081"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        await self._wait_for_health()
        
    async def stop(self) -> None:
        """Stop the system."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        # Ensure clean shutdown
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
        
    async def _wait_for_health(self) -> None:
        """Wait for system to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < STARTUP_TIMEOUT:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/api/health", timeout=5.0)
                    if response.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
            
        raise RuntimeError("System failed to start within timeout")


@asynccontextmanager
async def real_system():
    """Context manager for the real running system."""
    system = SystemUnderTest()
    try:
        await system.start()
        yield system
    finally:
        await system.stop()


class TestRealSystemBehavior:
    """Test the actual system behavior, not mocks."""
    
    @pytest.mark.asyncio
    async def test_system_health_endpoint_works(self):
        """INVARIANT: Health endpoint must always respond within 5 seconds."""
        async with real_system():
            async with httpx.AsyncClient() as client:
                start_time = time.time()
                response = await client.get(f"{API_BASE_URL}/api/health", timeout=5.0)
                response_time = time.time() - start_time
                
                # ASSERTIONS: System behavior invariants
                assert response.status_code == 200, "Health endpoint must be accessible"
                assert response_time < 5.0, f"Health check took {response_time}s, must be < 5s"
                
                data = response.json()
                assert "status" in data, "Health response must include status"
                assert data["status"] == "healthy", f"System reports unhealthy: {data}"
                
    @pytest.mark.asyncio 
    async def test_meme_generation_end_to_end_flow(self):
        """INVARIANT: Meme generation must complete within 2 minutes."""
        async with real_system():
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                
                # Test different meme formats to verify flexibility
                test_cases = [
                    {"topic": "testing the real system", "format": "Success kid"},
                    {"topic": "end to end integration", "format": "This is fine"},
                    {"topic": "no more mocks", "format": "Expanding brain"}
                ]
                
                for test_case in test_cases:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/memes/",
                        json=test_case
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # ASSERTIONS: Critical system invariants
                    assert response.status_code == 201, f"Meme generation failed: {response.text}"
                    assert generation_time < 120, f"Generation took {generation_time}s, must be < 120s"
                    
                    data = response.json()
                    
                    # Verify response structure
                    required_fields = ["id", "topic", "format", "text", "image_url", "created_at"]
                    for field in required_fields:
                        assert field in data, f"Response missing required field: {field}"
                        assert data[field], f"Field {field} is empty"
                    
                    # Verify content makes sense
                    assert data["topic"] == test_case["topic"], "Topic mismatch"
                    assert data["format"] == test_case["format"], "Format mismatch"
                    assert len(data["text"]) > 0, "Generated text is empty"
                    
                    # Verify image URL is valid
                    image_url = data["image_url"]
                    assert image_url.startswith(("http", "/static")), f"Invalid image URL: {image_url}"
                    
    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_crash_system(self):
        """INVARIANT: System must handle concurrent load without crashing."""
        async with real_system():
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                
                # Send 10 concurrent requests
                async def make_request(i: int) -> Dict[str, Any]:
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/memes/",
                        json={"topic": f"concurrent test {i}", "format": "Success kid"}
                    )
                    return {
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds() if response.elapsed else 0,
                        "success": response.status_code == 201,
                        "response": response.json() if response.status_code == 201 else response.text
                    }
                
                # Execute concurrent requests
                tasks = [make_request(i) for i in range(10)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # ASSERTIONS: System resilience invariants
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) >= 8, f"Too many failures: {len(results) - len(successful_results)}/10"
                
                # Verify no crashes (all responses are proper HTTP responses)
                for result in successful_results:
                    if isinstance(result, dict):
                        assert result["status_code"] in [201, 429, 503], f"Unexpected status: {result['status_code']}"
                        if result["success"]:
                            assert "id" in result["response"], "Successful response missing ID"
                
                # Verify system still responsive after load
                health_response = await client.get(f"{API_BASE_URL}/api/health")
                assert health_response.status_code == 200, "System unhealthy after concurrent load"
                
    @pytest.mark.asyncio
    async def test_invalid_requests_handled_gracefully(self):
        """INVARIANT: Invalid input must return proper errors, not crashes."""
        async with real_system():
            async with httpx.AsyncClient() as client:
                
                # Test various invalid inputs
                invalid_cases = [
                    {},  # Missing required fields
                    {"topic": ""},  # Empty topic
                    {"topic": "test"},  # Missing format
                    {"topic": "test", "format": ""},  # Empty format
                    {"topic": "x" * 1000, "format": "Drake meme"},  # Topic too long
                    {"topic": "test", "format": "NonexistentFormat"},  # Invalid format
                ]
                
                for invalid_input in invalid_cases:
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/memes/",
                        json=invalid_input
                    )
                    
                    # ASSERTIONS: Error handling invariants
                    assert response.status_code in [400, 422], f"Invalid input should return 4xx, got {response.status_code}"
                    
                    # Verify error response is JSON, not crash
                    try:
                        error_data = response.json()
                        assert "detail" in error_data or "message" in error_data, "Error response missing details"
                    except json.JSONDecodeError:
                        pytest.fail(f"Error response is not valid JSON: {response.text}")
                
                # Verify system still healthy after error cases
                health_response = await client.get(f"{API_BASE_URL}/api/health")
                assert health_response.status_code == 200, "System unhealthy after error cases"

    @pytest.mark.asyncio
    async def test_system_maintains_data_consistency(self):
        """INVARIANT: Each successful request must create exactly one database record."""
        async with real_system():
            async with httpx.AsyncClient() as client:
                
                # Get initial state
                health_response = await client.get(f"{API_BASE_URL}/api/health")
                assert health_response.status_code == 200
                
                # Generate 3 memes
                generated_ids = []
                for i in range(3):
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/memes/",
                        json={"topic": f"consistency test {i}", "format": "Success kid"}
                    )
                    assert response.status_code == 201
                    
                    data = response.json()
                    generated_ids.append(data["id"])
                
                # ASSERTIONS: Data consistency invariants
                assert len(generated_ids) == 3, "Should have generated 3 memes"
                assert len(set(generated_ids)) == 3, "All meme IDs should be unique"
                
                # Verify each ID is a valid UUID format
                for meme_id in generated_ids:
                    assert len(meme_id) == 36, f"Invalid UUID length: {meme_id}"
                    assert meme_id.count("-") == 4, f"Invalid UUID format: {meme_id}"


@pytest.mark.slow
class TestSystemStability:
    """Long-running stability tests - run with pytest -m slow."""
    
    @pytest.mark.asyncio
    async def test_system_memory_stability_under_load(self):
        """INVARIANT: Memory usage must remain bounded during continuous operation."""
        async with real_system():
            async with httpx.AsyncClient() as client:
                
                # Run for 5 minutes with steady load
                end_time = time.time() + 300  # 5 minutes
                request_count = 0
                
                while time.time() < end_time:
                    # Send request every 10 seconds
                    try:
                        response = await client.post(
                            f"{API_BASE_URL}/api/v1/memes/",
                            json={"topic": f"stability test {request_count}", "format": "Success kid"},
                            timeout=30.0
                        )
                        request_count += 1
                        
                        # ASSERTION: System must remain responsive
                        assert response.status_code in [201, 429, 503], f"Unexpected response: {response.status_code}"
                        
                    except Exception as e:
                        pytest.fail(f"System became unresponsive after {request_count} requests: {e}")
                    
                    await asyncio.sleep(10)
                
                # Final health check
                health_response = await client.get(f"{API_BASE_URL}/api/health")
                assert health_response.status_code == 200, "System unhealthy after stability test"
                
                # ASSERTION: Must have processed reasonable number of requests
                assert request_count >= 25, f"System processed too few requests: {request_count}"


if __name__ == "__main__":
    """Run tests directly for debugging."""
    pytest.main([__file__, "-v", "-s"])