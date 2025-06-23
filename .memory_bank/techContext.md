- **CLI (Potential):** Typer (>=0.9.0)
- **Configuration:** Pydantic Settings (`pydantic-settings>=2.1.0`) used in `config/config.py`.
- **Data Validation:** Pydantic (>=2.5.0)
- **Caching:** Redis (redis>=5.0.0), Aioredis (>=2.0.0) - Integrated via `api/dependencies.py` (`get_cache`).

## Development Tools

- **Logging:** Loguru (>=0.7.0), Structlog (>=24.1.0), Python standard `logging` (used in `main.py`).

## Infrastructure & Deployment (Inferred)

- **Environment Config:** `.env` file (sensitive keys, settings) - Parsed by Pydantic Settings in `config/config.py`.

## Key Implementation Files (Phase 1)

- **Main Entry Point:** `src/dspy_meme_gen/api/main.py`
- **Meme API Logic:** `src/dspy_meme_gen/api/routers/memes.py`
- **Health Check Logic:** `src/dspy_meme_gen/api/routers/health.py`
- **Configuration:** `src/dspy_meme_gen/config/config.py`
- **DB Connection:** `src/dspy_meme_gen/database/connection.py`
- **DB Models:** `src/dspy_meme_gen/models/database/memes.py`
- **API Schemas:** `src/dspy_meme_gen/models/schemas/memes.py`
- **DSPy Modules (Placeholders):** `src/dspy_meme_gen/dspy_modules/meme_predictor.py`, `src/dspy_meme_gen/dspy_modules/image_generator.py`
- **Cache Dependency:** `src/dspy_meme_gen/api/dependencies.py` (`get_cache`)
- **DB Session Dependency:** `src/dspy_meme_gen/database/connection.py` (`get_session`) 