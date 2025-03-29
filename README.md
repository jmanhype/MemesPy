# DSPy Meme Generator

A FastAPI-based meme generation service powered by DSPy for intelligent meme creation.

## Features

- Generate memes using AI with DSPy
- Analyze trending topics for meme creation
- Recommend suitable meme formats based on topics
- RESTful API for easy integration

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/dspy-meme-generator.git
   cd dspy-meme-generator
   ```
2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with the following content:

   ```
   # DSPy Configuration
   DSPY_MODEL_NAME=gpt-3.5-turbo-0125
   DSPY_TEMPERATURE=0.7
   DSPY_MAX_TOKENS=1024
   DSPY_RETRY_ATTEMPTS=2
   DSPY_RETRY_DELAY=1

   # API Keys
   OPENAI_API_KEY=your-openai-api-key-goes-here

   # Application Settings
   DEBUG=True
   ENVIRONMENT=development
   ```
5. Replace `your-openai-api-key-goes-here` with your actual OpenAI API key.

## Running the API

Start the API server with:

```
python -m uvicorn src.dspy_meme_gen.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000/`. The documentation is accessible at `http://127.0.0.1:8000/docs`.

## API Endpoints

### Memes

- `GET /api/v1/memes/` - List all generated memes
- `POST /api/v1/memes/` - Generate a new meme
- `GET /api/v1/memes/{meme_id}` - Get a specific meme

### Trends

- `GET /api/v1/trends/` - List trending topics
- `GET /api/v1/trends/{trend_id}` - Get a specific trending topic

### Formats

- `GET /api/v1/formats/` - List available meme formats
- `GET /api/v1/formats/{format_id}` - Get a specific meme format

## Example Usage

Generate a meme using curl:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memes/ \
  -H "Content-Type: application/json" \
  -d '{"topic": "Python Programming", "format": "standard"}'
```

## License

MIT

## Acknowledgements

- [DSPy](https://github.com/stanfordnlp/dspy) - For providing the foundation for LLM-based applications
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework
