# Meme Generation Actor System

This module implements an actor-based system for meme generation using the core actor framework. The system provides three main actors that work together to orchestrate the complete meme generation pipeline.

## Architecture Overview

The actor system follows a hierarchical design:

```
MemeGeneratorActor (Orchestrator)
├── TextGeneratorActor (DSPy Text Generation)
└── ImageGeneratorActor (DALL-E/gpt-image-1 Image Generation)
```

## Core Actors

### 1. MemeGeneratorActor

**Purpose**: Main orchestrator that coordinates the entire meme generation pipeline.

**Responsibilities**:
- Orchestrates text and image generation
- Handles quality scoring and refinement loops
- Manages state transitions through the generation process
- Provides metrics and monitoring

**Key Features**:
- State machine with clear progression through generation steps
- Circuit breakers for external service protection
- Configurable refinement loops with quality thresholds
- Comprehensive metrics collection

**Message Types**:
- `GenerateMemeRequest`: Start meme generation
- `GetMetricsRequest`: Retrieve performance metrics

### 2. TextGeneratorActor

**Purpose**: Handles DSPy-powered text generation for memes.

**Responsibilities**:
- Format selection and optimization
- Prompt generation for better DSPy results
- Text generation via DSPy modules
- Text verification and validation
- Caching of successful generations

**Key Features**:
- Smart caching to avoid redundant generations
- Format-aware prompt optimization
- Optional verification pipeline
- Circuit breaker protection for DSPy calls

**Message Types**:
- `GenerateTextRequest`: Generate meme text
- `VerifyTextRequest`: Verify generated text
- `ClearCacheRequest`: Clear generation cache

### 3. ImageGeneratorActor

**Purpose**: Handles image generation using OpenAI DALL-E/gpt-image-1.

**Responsibilities**:
- Image generation via OpenAI APIs
- Post-processing (text overlay, resizing, filters)
- CDN upload to Cloudinary
- Image caching and optimization

**Key Features**:
- Multiple image providers (OpenAI DALL-E)
- Automatic text overlay capabilities
- CDN integration with Cloudinary
- Comprehensive error handling and fallbacks

**Message Types**:
- `GenerateImageRequest`: Generate meme image
- `ProcessImageRequest`: Process existing images

## Usage Example

```python
import asyncio
from dspy_meme_gen.actors import (
    ActorSystem,
    MemeGeneratorActor,
    TextGeneratorActor, 
    ImageGeneratorActor,
    GenerateMemeRequest
)

async def generate_meme():
    # Create actor system
    system = ActorSystem("meme_generation")
    await system.start()
    
    try:
        # Spawn actors
        text_gen = await system.spawn(TextGeneratorActor, "text_generator")
        image_gen = await system.spawn(ImageGeneratorActor, "image_generator") 
        meme_gen = await system.spawn(
            MemeGeneratorActor,
            text_generator_ref=text_gen,
            image_generator_ref=image_gen,
            name="meme_orchestrator"
        )
        
        # Generate a meme
        request = GenerateMemeRequest(
            topic="working from home",
            format="drake",
            style="relatable",
            max_refinements=2,
            target_score=0.7
        )
        
        result = await meme_gen.ask(request, timeout=60000)
        print(f"Generated meme: {result}")
        
    finally:
        await system.stop()

# Run the example
asyncio.run(generate_meme())
```

## Configuration

### Environment Variables

The actors use these configuration settings from `settings`:

- `OPENAI_API_KEY`: Required for text and image generation
- `DSPY_MODEL_NAME`: DSPy model to use (default: gpt-3.5-turbo)
- `DSPY_TEMPERATURE`: Generation temperature
- `DSPY_MAX_TOKENS`: Maximum tokens for generation

### Circuit Breaker Settings

Each actor has configurable circuit breakers:

- **Text Generation**: 5 failures, 30s timeout
- **Image Generation**: 3 failures, 60s timeout  
- **CDN Upload**: 2 failures, 30s timeout

## State Management

### MemeGeneratorActor States

1. `IDLE` - Waiting for requests
2. `GENERATING_TEXT` - Text generation in progress
3. `SCORING` - Quality scoring in progress
4. `REFINING` - Text refinement in progress
5. `GENERATING_IMAGE` - Image generation in progress
6. `COMPLETED` - Generation completed successfully
7. `FAILED` - Generation failed

### TextGeneratorActor States

1. `IDLE` - Waiting for requests
2. `SELECTING_FORMAT` - Format selection in progress
3. `GENERATING_PROMPT` - Prompt optimization in progress
4. `GENERATING_TEXT` - DSPy text generation in progress
5. `VERIFYING` - Text verification in progress
6. `COMPLETED` - Generation completed
7. `FAILED` - Generation failed

### ImageGeneratorActor States

1. `IDLE` - Waiting for requests
2. `GENERATING_PROMPT` - Image prompt generation
3. `GENERATING_IMAGE` - DALL-E image generation
4. `POST_PROCESSING` - Image post-processing
5. `UPLOADING` - CDN upload in progress
6. `COMPLETED` - Generation completed
7. `FAILED` - Generation failed

## Error Handling

The actor system provides robust error handling:

1. **Circuit Breakers**: Protect against cascading failures from external services
2. **Timeouts**: Prevent hanging operations
3. **Graceful Degradation**: Fallback to simpler operations when advanced features fail
4. **Retry Logic**: Automatic retries with exponential backoff
5. **Comprehensive Logging**: Detailed error information for debugging

## Monitoring and Metrics

Each actor provides detailed metrics:

- **Performance**: Generation times, throughput
- **Quality**: Success rates, scores, refinement counts
- **Health**: Circuit breaker states, error rates
- **Cache**: Hit rates, cache sizes

## Integration with Existing DSPy Modules

The actors integrate seamlessly with existing DSPy agents:

- **MemeGenerator**: Core text generation
- **ScoringAgent**: Quality evaluation (simplified interface)
- **RefinementAgent**: Text improvement (simplified interface)
- **ImageRenderingAgent**: Image processing and CDN upload

## Future Enhancements

Planned improvements:

1. **Dynamic Actor Scaling**: Auto-scale actors based on load
2. **Advanced Scheduling**: Priority queues and load balancing
3. **Persistent State**: Actor state persistence across restarts
4. **Distributed Deployment**: Multi-node actor deployment
5. **Advanced Monitoring**: Prometheus metrics and Grafana dashboards