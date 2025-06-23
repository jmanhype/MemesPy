## Main User Flows

**Note:** The following describes the *intended* flow. As discovered in Phase 1, the current implementation uses placeholders for AI generation steps.

1.  **Generate Meme:**
    - User sends POST request to `/api/v1/memes/` with `topic` and `format`.
    - System checks cache (`get_cache` dependency, likely Redis) for existing meme based on topic/format.
    - If not cached:
        - **Current Behavior:** `routers/memes.py` calls `MemePredictor.forward()` and `ImageGenerator.generate()` from `dspy_modules/`. These methods **return hardcoded/random text and image URLs**, respectively.
        - **Intended Behavior:** `MemePredictor` would use DSPy and an LLM; `ImageGenerator` would use DALL-E or similar.
        - `routers/memes.py` saves the (placeholder) meme data to the database (via `get_session` dependency and `MemeDB` model).
        - Meme data is stored in the cache.
    - System returns the generated (placeholder) meme details (ID, text, image URL, etc.) as JSON.
// ... rest of flows ...

## AI Integration Details

- **DSPy Modules:** `meme_predictor.py` and `image_generator.py` contain the *intended* structure for AI logic.
- **Current Status:** Both `MemePredictor.forward` and `ImageGenerator.generate` methods used by the API router (`routers/memes.py`) are **placeholders** and do not perform actual AI/API calls. They return random, predefined content.
- **Signature:** `MemeSignature` is defined in `meme_predictor.py` but not used by the current placeholder `MemePredictor.forward` method.
- **Configuration:** DSPy settings (model name, temperature, tokens) are defined in `config/config.py` and loaded from `.env`, but are not actively used by the placeholder implementation.
- **Unused Module:** `TrendPredictor` exists in `meme_predictor.py` and demonstrates actual DSPy `ChainOfThought` usage, but it's not integrated into the main API flow. 