# Comprehensive Testing Strategy for Actor-Based MemesPy System

This directory contains a comprehensive testing strategy designed to verify the resilience and reliability of the MemesPy system under various failure scenarios and edge cases.

## Testing Categories

### 1. Property-Based Testing with Hypothesis (`test_actors_hypothesis.py`)

Property-based testing ensures that system invariants hold regardless of input:

- **Actor Invariants**: Tests that all actors maintain their contracts
- **Pipeline Properties**: Verifies system-wide properties like determinism
- **State Machine Testing**: Models the system as a state machine to find edge cases
- **Input Fuzzing**: Tests with randomly generated, potentially malicious inputs

Key features:
- Custom strategies for generating valid and invalid meme requests
- Stateful testing to catch sequence-dependent bugs
- Performance regression detection
- Memory leak detection

### 2. Chaos Engineering (`test_chaos_engineering.py`)

Chaos engineering systematically injects failures to test resilience:

- **Network Chaos**: Simulates timeouts, connection failures, DNS issues
- **Database Chaos**: Tests connection pool exhaustion, deadlocks, corruption
- **Memory Chaos**: Simulates memory leaks and out-of-memory conditions
- **Concurrency Chaos**: Tests race conditions and deadlocks
- **Cache Chaos**: Tests cache inconsistency and avalanche scenarios
- **Time-Based Chaos**: Tests clock skew and timeout cascades

Key features:
- ChaosMonkey class for controlled failure injection
- Configurable failure rates and types
- Detailed failure tracking and reporting

### 3. Load Testing and Performance Benchmarks (`test_load_performance.py`)

Comprehensive load testing to measure system performance:

- **Steady Load Testing**: Tests system under constant load
- **Spike Load Testing**: Tests sudden traffic increases
- **Sustained High Load**: Tests extended periods of high traffic
- **Concurrent User Scaling**: Tests how system scales with users
- **Performance Benchmarking**: Measures individual component performance

Key features:
- Detailed performance metrics (P50, P95, P99 latencies)
- Resource usage monitoring (CPU, memory)
- Throughput measurement
- Visual reporting with graphs
- Locust integration for realistic load patterns

### 4. Contract Testing (`test_service_contracts.py`)

Ensures services maintain their API contracts:

- **Schema Validation**: Validates input/output against JSON schemas
- **Service Compatibility**: Tests that services can communicate
- **Contract Evolution**: Tests backward compatibility
- **Consumer-Driven Contracts**: Uses Pact for CDC testing

Key features:
- Comprehensive contract schemas for all services
- Automatic contract violation detection
- Contract monitoring for production
- Compatibility matrix generation

### 5. Fault Injection Testing (`test_fault_injection.py`)

Systematic fault injection framework:

- **Exception Injection**: Injects various exception types
- **Delay Injection**: Adds latency to operations
- **Data Corruption**: Corrupts data in transit
- **Resource Exhaustion**: Exhausts system resources
- **Signal Injection**: Tests signal handling

Key features:
- Configurable fault scenarios
- Probability-based injection
- Fault orchestration for complex scenarios
- Production-ready fault injection decorators

### 6. Integration Resilience Testing (`test_integration_resilience.py`)

Tests resilience with real external services:

- **Database Failover**: Tests behavior during database outages
- **Cache Degradation**: Tests Redis failures and recovery
- **API Rate Limiting**: Tests rate limit handling
- **Network Partitions**: Tests network isolation scenarios
- **Cascading Failures**: Tests multi-service failures
- **24-Hour Stability**: Tests long-term stability
- **Deployment Scenarios**: Tests zero-downtime deployments
- **Disaster Recovery**: Tests backup and restore procedures

Key features:
- Docker integration for real service testing
- Service health monitoring
- Recovery time measurement
- Availability calculation

## Running the Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Additional dependencies for comprehensive testing
pip install hypothesis locust pytest-asyncio testcontainers docker
```

### Running Individual Test Suites

```bash
# Property-based testing
pytest tests/property_based/test_actors_hypothesis.py -v

# Chaos engineering (requires elevated permissions for some tests)
pytest tests/chaos/test_chaos_engineering.py -v

# Load testing
pytest tests/load_testing/test_load_performance.py -v -m "not slow"

# Contract testing
pytest tests/contracts/test_service_contracts.py -v

# Fault injection
pytest tests/fault_injection/test_fault_injection.py -v

# Integration resilience (requires Docker)
pytest tests/resilience/test_integration_resilience.py -v -m integration
```

### Running Comprehensive Test Suite

```bash
# Run all resilience tests
pytest tests/resilience/ -v

# Run with specific markers
pytest tests/ -m "not slow and not integration" -v

# Run with coverage
pytest tests/resilience/ --cov=src/dspy_meme_gen --cov-report=html
```

### Running Load Tests with Locust

```bash
# Start Locust web UI
locust -f tests/load_testing/test_load_performance.py --host=http://localhost:8081

# Run headless
locust -f tests/load_testing/test_load_performance.py \
  --host=http://localhost:8081 \
  --users=100 \
  --spawn-rate=10 \
  --run-time=5m \
  --headless
```

## Test Configuration

### Environment Variables

```bash
# Core services
export DATABASE_URL=postgresql://user:pass@localhost:5432/memes_test
export REDIS_URL=redis://localhost:6379
export OPENAI_API_KEY=your-api-key

# Test configuration
export SKIP_DOCKER_SETUP=1  # Skip Docker setup if services already running
export TEST_DURATION=3600   # Duration for long-running tests
export CHAOS_FAILURE_RATE=0.3  # Default failure rate for chaos tests
```

### Docker Compose for Testing

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: memes_test
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  app:
    build: .
    environment:
      DATABASE_URL: postgresql://test:test@postgres:5432/memes_test
      REDIS_URL: redis://redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - postgres
      - redis
```

## Interpreting Test Results

### Success Criteria

1. **Property-Based Tests**: All properties hold for 100+ examples
2. **Chaos Tests**: System maintains >80% success rate under failures
3. **Load Tests**: 
   - P95 latency < 5 seconds
   - Success rate > 95%
   - No memory leaks
4. **Contract Tests**: 100% contract compliance
5. **Fault Injection**: Graceful degradation, no cascading failures
6. **Integration Tests**: 
   - Recovery time < 30 seconds
   - Availability > 99.9%

### Performance Baselines

Expected performance under normal conditions:
- Response time: P50 < 1s, P95 < 3s, P99 < 5s
- Throughput: > 50 requests/second
- Error rate: < 1%
- CPU usage: < 80% under load
- Memory usage: < 1GB per instance

### Failure Handling

Expected behavior during failures:
- Database outage: Fallback to cache, degraded functionality
- Cache failure: Direct database access, slower but functional
- API failures: Retry with exponential backoff, circuit breaking
- High load: Rate limiting, queue management
- Memory pressure: Graceful degradation, no OOM kills

## Continuous Testing

### CI/CD Integration

```yaml
# .github/workflows/resilience-tests.yml
name: Resilience Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'

jobs:
  resilience-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run property-based tests
        run: pytest tests/property_based/ -v
      
      - name: Run chaos tests
        run: pytest tests/chaos/ -v -m "not slow"
      
      - name: Run contract tests
        run: pytest tests/contracts/ -v
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            htmlcov/
            test-reports/
            *.png
            *.json
```

### Monitoring in Production

The test suite includes monitoring capabilities that can be used in production:

1. **Contract Monitoring**: Validates API contracts in real-time
2. **Chaos Engineering**: Can be enabled with reduced failure rates
3. **Performance Benchmarks**: Regular performance regression detection
4. **Health Checks**: Comprehensive service health monitoring

## Best Practices

1. **Run property-based tests frequently** - They find edge cases
2. **Use chaos engineering in staging** - Test resilience before production
3. **Monitor contract compliance** - Catch breaking changes early
4. **Benchmark performance regularly** - Detect regressions
5. **Test disaster recovery quarterly** - Ensure procedures work
6. **Update test scenarios** - Based on production incidents

## Troubleshooting

### Common Issues

1. **Docker connection errors**: Ensure Docker daemon is running
2. **Database connection failures**: Check PostgreSQL is accessible
3. **Redis connection issues**: Verify Redis is running
4. **API rate limits**: Use test API keys with higher limits
5. **Memory issues**: Increase Docker memory allocation

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export PYTEST_VERBOSE=2

# Run with full output
pytest tests/resilience/ -vvs --tb=long
```

## Contributing

When adding new tests:

1. Follow the established patterns in each test category
2. Add appropriate markers (`@pytest.mark.slow`, `@pytest.mark.integration`)
3. Document expected behavior and success criteria
4. Include cleanup code to prevent test pollution
5. Add configuration options for CI/CD compatibility

## References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
- [Locust Load Testing](https://docs.locust.io/)
- [Pact Contract Testing](https://docs.pact.io/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)