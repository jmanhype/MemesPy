"""Performance and load tests for the meme generation pipeline."""
from typing import Dict, List, Optional
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.dspy_meme_gen.models.repository import MemeTemplateRepository
from src.dspy_meme_gen.models.meme import MemeTemplate
from tests.config import test_config
from tests.utils.performance import (
    PerformanceMetrics,
    measure_performance,
    run_concurrent_operations
)
from tests.utils.load import (
    LoadTestMetrics,
    run_load_test,
    MemeGeneratorUser
)
from tests.utils.mocks import (
    MockOpenAI,
    MockCloudinary,
    MockRedis
)

@pytest.mark.asyncio
async def test_meme_template_repository_performance(
    load_test_db: AsyncSession,
    performance_metrics: Dict[str, PerformanceMetrics]
) -> None:
    """Test MemeTemplateRepository performance.
    
    Args:
        load_test_db: Database session with performance monitoring
        performance_metrics: Performance metrics dictionary
    """
    repository = MemeTemplateRepository(load_test_db)
    
    # Create test data
    templates = [
        MemeTemplate(
            name=f"template_{i}",
            description=f"Test template {i}",
            format_type="image",
            structure={"text": ["top", "bottom"]},
            popularity_score=1.0
        )
        for i in range(100)
    ]
    
    # Test bulk insert performance
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def bulk_insert() -> None:
        await repository.bulk_create(templates)
    
    await bulk_insert()
    
    # Test query performance
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def query_templates() -> List[MemeTemplate]:
        return await repository.get_all(
            limit=10,
            offset=0,
            order_by="popularity_score"
        )
    
    await query_templates()
    
    # Test search performance
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def search_templates() -> List[MemeTemplate]:
        return await repository.search(
            query="test",
            fields=["name", "description"]
        )
    
    await search_templates()

@pytest.mark.asyncio
async def test_concurrent_database_operations(
    load_test_db: AsyncSession,
    performance_metrics: Dict[str, PerformanceMetrics]
) -> None:
    """Test concurrent database operations.
    
    Args:
        load_test_db: Database session with performance monitoring
        performance_metrics: Performance metrics dictionary
    """
    repository = MemeTemplateRepository(load_test_db)
    
    # Test concurrent reads
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def read_template(template_id: int) -> Optional[MemeTemplate]:
        return await repository.get_by_id(template_id)
    
    results = await run_concurrent_operations(
        read_template,
        num_concurrent=test_config.LOAD_TEST_CONCURRENT_USERS,
        template_id=1
    )
    
    assert len(results) == test_config.LOAD_TEST_CONCURRENT_USERS
    
    # Test concurrent writes
    @measure_performance(threshold_ms=test_config.MAX_QUERY_TIME_MS)
    async def create_template() -> MemeTemplate:
        template = MemeTemplate(
            name="concurrent_test",
            description="Test template",
            format_type="image",
            structure={"text": ["top", "bottom"]},
            popularity_score=1.0
        )
        return await repository.create(template)
    
    results = await run_concurrent_operations(
        create_template,
        num_concurrent=test_config.LOAD_TEST_CONCURRENT_USERS
    )
    
    assert len(results) == test_config.LOAD_TEST_CONCURRENT_USERS

@pytest.mark.asyncio
async def test_cache_performance(
    mock_redis: MockRedis,
    performance_metrics: Dict[str, PerformanceMetrics]
) -> None:
    """Test cache performance.
    
    Args:
        mock_redis: Mock Redis client
        performance_metrics: Performance metrics dictionary
    """
    # Test cache write performance
    @measure_performance(threshold_ms=test_config.MAX_CACHE_TIME_MS)
    async def write_to_cache(key: str, value: bytes) -> None:
        await mock_redis.set(key, value)
    
    await write_to_cache("test_key", b"test_value")
    
    # Test cache read performance
    @measure_performance(threshold_ms=test_config.MAX_CACHE_TIME_MS)
    async def read_from_cache(key: str) -> Optional[bytes]:
        return await mock_redis.get(key)
    
    result = await read_from_cache("test_key")
    assert result == b"test_value"

@pytest.mark.asyncio
async def test_meme_generation_load(
    load_test_db: AsyncSession,
    mock_openai: MockOpenAI,
    mock_cloudinary: MockCloudinary,
    mock_redis: MockRedis,
    load_test_metrics: LoadTestMetrics
) -> None:
    """Test meme generation under load.
    
    Args:
        load_test_db: Database session with performance monitoring
        mock_openai: Mock OpenAI client
        mock_cloudinary: Mock Cloudinary client
        mock_redis: Mock Redis client
        load_test_metrics: Load test metrics
    """
    # Test meme generation performance
    @measure_performance(threshold_ms=test_config.MAX_RESPONSE_TIME_MS)
    async def generate_meme() -> Dict[str, Any]:
        # Mock meme generation process
        # 1. Get template from database
        repository = MemeTemplateRepository(load_test_db)
        template = await repository.get_by_id(1)
        
        # 2. Generate image with OpenAI
        image_result = await mock_openai.create_image(
            prompt="Test meme prompt"
        )
        
        # 3. Upload to Cloudinary
        upload_result = mock_cloudinary.upload(
            image_result["data"][0]["url"]
        )
        
        # 4. Cache result
        await mock_redis.set(
            f"meme:1",
            str(upload_result["secure_url"]).encode()
        )
        
        return {
            "template": template,
            "image_url": upload_result["secure_url"]
        }
    
    # Run load test
    metrics = await run_load_test(
        generate_meme,
        concurrent_users=test_config.LOAD_TEST_CONCURRENT_USERS,
        duration_seconds=test_config.LOAD_TEST_DURATION_SECONDS,
        spawn_rate=test_config.LOAD_TEST_SPAWN_RATE
    )
    
    # Update load test metrics
    load_test_metrics.total_requests = metrics.total_requests
    load_test_metrics.successful_requests = metrics.successful_requests
    load_test_metrics.failed_requests = metrics.failed_requests
    load_test_metrics.response_times = metrics.response_times
    load_test_metrics.errors = metrics.errors
    
    # Assert performance requirements
    assert metrics.success_rate >= 95.0, "Success rate below 95%"
    assert metrics.avg_response_time <= test_config.MAX_RESPONSE_TIME_MS, (
        f"Average response time {metrics.avg_response_time:.2f}ms exceeds "
        f"threshold of {test_config.MAX_RESPONSE_TIME_MS}ms"
    ) 