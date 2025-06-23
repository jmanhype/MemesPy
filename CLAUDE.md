# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
# Start the API server
python -m uvicorn src.dspy_meme_gen.api.main:app --port 8081

# Start with hot reload for development
python -m uvicorn src.dspy_meme_gen.api.main:app --reload --port 8081
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m api          # API tests only

# Run a single test file
pytest tests/test_specific_file.py

# Run with coverage report
pytest --cov=dspy_meme_gen --cov-report=term-missing
```

### Code Quality
```bash
# Run linting
ruff check src/ tests/

# Run type checking
mypy src/

# Format code (ruff uses black-compatible formatting)
ruff format src/ tests/
```

### Database Management
```bash
# Initialize database tables
python scripts/init_db.py

# Run database migrations (when using Alembic)
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "Description of changes"
```

### Docker Operations
```bash
# Build and start all services
docker-compose up --build

# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## High-Level Architecture

### Core Architecture Pattern
The application follows a clean architecture pattern with clear separation of concerns:

1. **API Layer** (`src/dspy_meme_gen/api/`): FastAPI endpoints handling HTTP requests
2. **Service Layer** (`src/dspy_meme_gen/services/`): Business logic and orchestration
3. **Agent Layer** (`src/dspy_meme_gen/agents/`): DSPy-powered AI agents for various tasks
4. **Data Layer** (`src/dspy_meme_gen/database/`, `models/`): Database access and models

### Key Architectural Decisions

#### DSPy Integration
- All AI functionality uses DSPy modules with declarative signatures
- Agents are modular and can be composed into pipelines
- Each agent has a specific responsibility (routing, verification, generation, etc.)

#### Async-First Design
- FastAPI with async/await throughout
- Async database operations with SQLAlchemy async
- Redis caching with aioredis for performance

#### Verification Pipeline
The system implements a multi-stage verification process:
1. **Router Agent**: Analyzes requests and determines verification needs
2. **Verification Agents**: Check factuality, appropriateness, and instruction following
3. **Scoring Agent**: Evaluates meme quality
4. **Refinement Loop**: Iteratively improves low-scoring memes

#### Monitoring and Observability
- Prometheus metrics for performance tracking
- Grafana dashboards for visualization
- Comprehensive logging with correlation IDs
- Health check endpoints for service monitoring

### Critical Dependencies
- **DSPy**: Core AI framework - all meme generation logic depends on this
- **FastAPI**: Web framework - all API endpoints
- **SQLAlchemy**: ORM for database operations
- **Redis**: Caching layer for performance
- **Pillow**: Image processing and manipulation
- **OpenAI API**: LLM provider for DSPy

### Environment Configuration
The application relies heavily on environment variables. Key required settings:
- `OPENAI_API_KEY`: Required for DSPy/OpenAI integration
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection for caching
- `DSPY_MODEL_NAME`: Model to use (e.g., gpt-3.5-turbo)

### Error Handling Strategy
- Custom exception hierarchy rooted at `DSPyMemeError`
- Specific exceptions for validation, content, generation, and external service errors
- All exceptions include detailed error codes and messages for debugging