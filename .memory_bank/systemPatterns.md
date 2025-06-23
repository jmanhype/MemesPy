- **Modular Design:** Code is organized into modules based on functionality (e.g., `cache`, `database`, `config`, `utils`, `exceptions`, `health`, `monitoring`).
- **Entry Point:** `src/dspy_meme_gen/api/main.py` initializes FastAPI and includes routers (`routers/health.py`, `routers/memes.py`).

## Key Design Patterns

- **Dependency Injection:** Used heavily by FastAPI. Key dependencies identified:
    - `get_session` (from `database/connection.py`) for SQLAlchemy sessions.
    - `get_cache` (from `api/dependencies.py`) for Redis cache connection.
- **Configuration Management:** Uses Pydantic `BaseSettings` in `config/config.py` loading from `.env` file.

## AI/ML Integration

- **DSPy Orchestration:** Core logic for generating meme text and image prompts is handled by modules in `dspy_modules/`.
    - `MemePredictor`: **Currently a placeholder**, returns random hardcoded text. Defines `MemeSignature` (unused by placeholder).
    - `ImageGenerator`: **Currently a placeholder**, returns random hardcoded image URLs.
    - `TrendPredictor`: Shows example DSPy usage but is not integrated into the main API flow.
- **DSPy Configuration:** `ensure_dspy_configured()` utility exists in `meme_predictor.py` to set up `dspy.settings.lm` using `config.settings`.
- **Integration Point:** The API router `routers/memes.py` calls the (placeholder) `MemePredictor` and `ImageGenerator` instances. 