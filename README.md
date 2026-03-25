# MemesPy

FastAPI service that generates memes using DSPy for text and OpenAI for images.

## How It Works

1. User submits a topic and optional format via the REST API.
2. DSPy (ChainOfThought) generates meme text and an image prompt.
3. Image is generated via gpt-image-1 (saved locally) or DALL-E 3 (URL returned).
4. Meme is stored in SQLite with metadata (generation time, cost, quality scores).

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI, served by Uvicorn |
| Text generation | DSPy with OpenAI models |
| Image generation | gpt-image-1 (primary), DALL-E 3 (fallback) |
| Database | SQLAlchemy + SQLite |
| Cache | Redis (optional) |
| Monitoring | Grafana dashboards, Prometheus |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/v1/memes/` | Generate a meme |
| GET | `/api/v1/memes/` | List memes |
| GET | `/api/v1/memes/{id}` | Get a specific meme |
| GET | `/api/v1/analytics/stats/{period}` | Generation statistics |
| GET | `/api/v1/analytics/trending` | Trending memes |
| POST | `/api/v1/analytics/search` | Search by metadata |

## Requirements

- Python 3.10+
- OpenAI API key

## Setup

```bash
git clone https://github.com/jmanhype/MemesPy.git
cd MemesPy
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python scripts/init_db.py
python -m uvicorn src.dspy_meme_gen.api.main:app --port 8081
```

API docs at `localhost:8081/docs`.

## Tests

```bash
pytest
```

Test suite includes unit tests for agents (factuality, image renderer, prompt generator, router, scorer), integration tests, and actor supervision tests.

## Status

Working prototype. Meme text generation and image generation function correctly with valid API keys. The analytics and metadata tracking endpoints are built. The repository contains 40+ generated meme PNGs committed to `static/images/memes/` which inflates the repo size. The articles directory contains draft content for a blog post about the project.

## License

AGPL-3.0 (core). Commercial components under `platform/` are proprietary (directory does not yet exist).
