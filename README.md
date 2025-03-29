# DSPy Meme Generator

A FastAPI-based meme generation service powered by DSPy for intelligent meme creation.

## Features

- Generate memes using AI with DSPy
- Analyze trending topics for meme creation
- Recommend suitable meme formats based on topics
- RESTful API for easy integration
- SQLite database for persistent storage
- Simple caching mechanism for improved performance

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
   DATABASE_URL=sqlite:///./meme_generator.db
   CACHE_TTL=3600
   ```
5. Replace `your-openai-api-key-goes-here` with your actual OpenAI API key.

## Running the API

Start the API server with:

```
python -m uvicorn src.dspy_meme_gen.api.final_app:app --port 8081
```

The API will be available at `http://127.0.0.1:8081/`. The documentation is accessible at `http://127.0.0.1:8081/docs`.

## Testing the Image Generator

You can test the image generation capabilities using the provided script:

```bash
python scripts/test_image_generator.py
```

This script demonstrates three different image generation approaches:

1. **Placeholder**: Returns sample image URLs (for testing without API keys)
2. **DALL-E**: Uses OpenAI's DALL-E model for image generation
3. **GPT-4o**: A placeholder implementation for the upcoming GPT-4o image generation API

The script displays the results in a table format, making it easy to compare the different providers.

## API Endpoints

### Health Check

- `GET /api/health` - Check the health of the API

Example:

```bash
curl http://127.0.0.1:8081/api/health
```

Response:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "openai_configured": true,
  "dspy_configured": true
}
```

### Memes

- `GET /api/v1/memes/` - List all generated memes
- `POST /api/v1/memes/` - Generate a new meme
- `GET /api/v1/memes/{meme_id}` - Get a specific meme

### Example: Generate a Meme

```bash
curl -X POST http://127.0.0.1:8081/api/v1/memes/ \
  -H "Content-Type: application/json" \
  -d '{"topic": "Python Programming", "format": "standard"}'
```

Response:

```json
{
  "id": "b1691d8c-c16d-4128-bf29-010557116f1c",
  "topic": "Python Programming",
  "format": "standard",
  "text": "Sample meme text about Python Programming",
  "image_url": "https://example.com/sample.jpg",
  "created_at": "2025-03-28T18:15:35.647677",
  "score": 0.95
}
```

### Example: List All Memes

```bash
curl http://127.0.0.1:8081/api/v1/memes/
```

Response:

```json
{
  "items": [
    {
      "id": "b1691d8c-c16d-4128-bf29-010557116f1c",
      "topic": "Python Programming",
      "format": "standard",
      "text": "Sample meme text about Python Programming",
      "image_url": "https://example.com/sample.jpg",
      "created_at": "2025-03-28T18:15:35.647677",
      "score": 0.95
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### Example: Get a Specific Meme

```bash
curl http://127.0.0.1:8081/api/v1/memes/b1691d8c-c16d-4128-bf29-010557116f1c
```

Response:

```json
{
  "id": "b1691d8c-c16d-4128-bf29-010557116f1c",
  "topic": "Python Programming",
  "format": "standard",
  "text": "Sample meme text about Python Programming",
  "image_url": "https://example.com/sample.jpg",
  "created_at": "2025-03-28T18:15:35.647677",
  "score": 0.95
}
```

## Working with DSPy

The meme generator uses DSPy to create the meme content, leveraging the power of large language models. DSPy modules are implemented in the `src/dspy_meme_gen/dspy_modules/` directory:

- `meme_predictor.py` - Uses DSPy to generate the text content for the meme
- `image_generator.py` - Generates an image URL based on the meme content

When a meme is requested, the following happens:

1. The app checks if a meme with the same topic and format exists in the cache
2. If not found in cache, DSPy is used to generate meme text content
3. An image URL is generated (or a placeholder is used in demo mode)
4. The meme is stored in both the database and cache
5. The meme is returned to the user

### DSPy Module Implementation

Our project uses DSPy's signature-based programming model to create AI-powered meme content. Here's a breakdown of our key DSPy components:

#### MemeSignature

```python
class MemeSignature(dspy.Signature):
    """Signature for meme generation."""
  
    topic: str = dspy.InputField(desc="The topic or theme for the meme")
    format: str = dspy.InputField(desc="The meme format to use (e.g., 'standard', 'modern', 'comparison')")
    context: Optional[str] = dspy.InputField(desc="Additional context or requirements for the meme")
  
    text: str = dspy.OutputField(desc="The text content for the meme")
    image_prompt: str = dspy.OutputField(desc="A detailed prompt for image generation")
    rationale: str = dspy.OutputField(desc="Explanation of why this meme would be effective")
    score: float = dspy.OutputField(desc="A score between 0 and 1 indicating the predicted effectiveness")
```

This signature defines the input and output fields for our meme generation process. DSPy uses this signature to generate appropriate text and image prompts.

#### DSPy Configuration

```python
def ensure_dspy_configured() -> None:
    """Ensure DSPy is configured with a language model."""
    try:
        # Check if DSPy is already configured
        _ = dspy.settings.lm
    except AttributeError:
        # Configure DSPy with LM using the correct API
        lm = dspy.LM(
            f"openai/{settings.dspy_model_name}",
            api_key=settings.openai_api_key
        )
        dspy.configure(lm=lm)
```

This function ensures that DSPy is properly configured with the OpenAI language model before we try to use it.

#### MemePredictor

The MemePredictor class uses DSPy's ChainOfThought to generate creative meme content based on the provided topic and format. It includes fallback mechanisms for when DSPy is not properly configured.

#### Advanced DSPy Features

For future versions, we plan to incorporate more advanced DSPy features:

- DSPy's teleprompter for optimizing meme generation
- Multi-stage refinement using DSPy's pipeline capabilities
- Trainable modules that learn from user feedback to improve meme quality over time

### Using DSPy Directly

You can also use the DSPy modules directly in your own Python scripts. Here's an example:

```python
import os
import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure DSPy with OpenAI
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("DSPY_MODEL_NAME", "gpt-3.5-turbo-0125")
lm = dspy.LM(f"openai/{model_name}", api_key=api_key)
dspy.configure(lm=lm)

# Define a signature for meme generation
class MemeSignature(dspy.Signature):
    topic: str = dspy.InputField()
    format: str = dspy.InputField()
  
    text: str = dspy.OutputField()
    image_prompt: str = dspy.OutputField()
    score: float = dspy.OutputField()

# Create a module using ChainOfThought
generate_meme = dspy.ChainOfThought(MemeSignature)

# Generate a meme
result = generate_meme(
    topic="Python Programming",
    format="standard"
)

# Print the results
print(f"Meme text: {result.text}")
print(f"Image prompt: {result.image_prompt}")
print(f"Quality score: {result.score}")
```

This script configures DSPy, defines a signature, creates a module, and generates a meme directly using the DSPy framework.

## Future Enhancements: GPT-4o Image Generation

We're excited about the upcoming integration with OpenAI's GPT-4o image generation capabilities! As announced by OpenAI in March 2025, GPT-4o offers native image generation directly from the model that will soon be available through the API.

### Key Features of GPT-4o Image Generation

- **Native integration with text generation**: Both text and images can be generated by the same model, providing superior context awareness
- **Text rendering excellence**: GPT-4o excels at accurately embedding text within images, perfect for memes that combine visuals and text
- **Multi-turn refinement**: Images can be iteratively improved through conversation, allowing for precise adjustments
- **Contextual awareness**: The model can use previous images and text exchanges to maintain visual consistency
- **High-quality visuals**: Supports photorealistic images and various artistic styles

### Implementation Plan

Our meme generator is perfectly positioned to leverage GPT-4o's image generation capabilities once the API is released. The planned implementation includes:

1. **Seamless integration**: Our modular architecture will allow easy integration of GPT-4o image generation
2. **Context-aware generation**: Utilizing both the meme text and prompt in a single API call
3. **Style customization**: Allowing users to specify meme styles, from photorealistic to cartoon
4. **Image refinement**: Supporting iterative improvement of generated memes

### Benefits for Meme Generation

GPT-4o's capabilities will dramatically enhance our meme generator:

- **Text-in-image quality**: Memes often require clear text overlaid on images, which GPT-4o handles exceptionally well
- **Higher quality images**: More realistic, detailed, and creative visuals
- **Better theme adherence**: Images more accurately reflecting the specified meme topic
- **Multi-object scenes**: Ability to position multiple elements within a meme exactly as described

### Timeline

As OpenAI has announced that the GPT-4o image generation API will be available "in the coming weeks," we're preparing our codebase now to quickly implement this feature as soon as the API is released.

No code changes are needed to start experimenting with this today - our architecture is designed to easily swap in new image generation providers, and all API keys can be configured through environment variables.

## Development

### Project Structure

```
dspy-meme-generator/
├── src/
│   └── dspy_meme_gen/
│       ├── api/
│       │   ├── dependencies.py
│       │   ├── final_app.py
│       │   ├── main.py
│       │   └── routers/
│       │       ├── memes.py
│       │       └── health.py
│       ├── config/
│       │   └── settings.py
│       ├── database/
│       │   └── connection.py
│       ├── dspy_modules/
│       │   ├── meme_predictor.py
│       │   └── image_generator.py
│       └── models/
│           ├── database/
│           │   └── memes.py
│           └── schemas/
│               └── memes.py
├── scripts/
│   └── test_image_generator.py
├── .env
├── requirements.txt
└── README.md
```

### Testing

Run the tests with:

```
pytest
```

## License

MIT

## Acknowledgements

- [DSPy](https://github.com/stanfordnlp/dspy) - For providing the foundation for LLM-based applications
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - For ORM database interactions
