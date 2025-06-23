"""Example usage of the meme generation actors."""

import asyncio
import logging
from typing import Dict, Any

from .core import ActorSystem
from .meme_generator_actor import (
    MemeGeneratorActor, 
    GenerateMemeRequest,
    GetMetricsRequest
)
from .text_generator_actor import (
    TextGeneratorActor,
    GenerateTextRequest,
    ClearCacheRequest
)
from .image_generator_actor import (
    ImageGeneratorActor,
    GenerateImageRequest,
    ProcessImageRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_actor_system():
    """Demonstrate the meme generation actor system."""
    
    # Create actor system
    system = ActorSystem("meme_generation_system")
    await system.start()
    
    try:
        # Spawn actors
        logger.info("Creating actors...")
        
        # Create individual actors
        text_gen_ref = await system.spawn(TextGeneratorActor, "text_generator")
        image_gen_ref = await system.spawn(ImageGeneratorActor, "image_generator")
        
        # Create main orchestrator with references to other actors
        meme_gen_ref = await system.spawn(
            MemeGeneratorActor,
            text_generator_ref=text_gen_ref,
            image_generator_ref=image_gen_ref,
            name="meme_orchestrator"
        )
        
        logger.info("All actors created successfully")
        
        # Test individual actors
        await test_text_generation(text_gen_ref)
        await test_image_generation(image_gen_ref)
        
        # Test the full meme generation pipeline
        await test_full_meme_generation(meme_gen_ref)
        
        # Show metrics
        await show_metrics(meme_gen_ref, text_gen_ref, image_gen_ref)
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise
    finally:
        # Clean shutdown
        logger.info("Shutting down actor system...")
        await system.stop()
        logger.info("Actor system shutdown complete")


async def test_text_generation(text_gen_ref):
    """Test text generation actor."""
    logger.info("=== Testing Text Generation Actor ===")
    
    try:
        # Test basic text generation
        request = GenerateTextRequest(
            topic="cats vs dogs",
            format="drake",
            style="humorous"
        )
        
        logger.info(f"Requesting text generation for: {request.topic}")
        response = await text_gen_ref.ask(request, timeout=15000)
        
        logger.info(f"Generated text: {response.get('text', 'N/A')}")
        logger.info(f"Generation time: {response.get('generation_time', 0):.2f}s")
        logger.info(f"Verified: {response.get('verified', False)}")
        
    except Exception as e:
        logger.error(f"Text generation test failed: {e}")


async def test_image_generation(image_gen_ref):
    """Test image generation actor."""
    logger.info("=== Testing Image Generation Actor ===")
    
    try:
        # Test basic image generation
        request = GenerateImageRequest(
            text="When you see a good meme format",
            format="drake",
            style="modern",
            size="512x512",
            upload_to_cdn=False  # Skip CDN for demo
        )
        
        logger.info(f"Requesting image generation for: {request.text}")
        response = await image_gen_ref.ask(request, timeout=30000)
        
        logger.info(f"Generated image URL: {response.get('image_url', 'N/A')[:100]}...")
        logger.info(f"Generation time: {response.get('generation_time', 0):.2f}s")
        logger.info(f"CDN uploaded: {response.get('cdn_uploaded', False)}")
        
    except Exception as e:
        logger.error(f"Image generation test failed: {e}")


async def test_full_meme_generation(meme_gen_ref):
    """Test the full meme generation pipeline."""
    logger.info("=== Testing Full Meme Generation Pipeline ===")
    
    try:
        # Test end-to-end meme generation
        request = GenerateMemeRequest(
            topic="working from home vs office",
            format="distracted_boyfriend",
            style="relatable",
            max_refinements=2,
            target_score=0.7
        )
        
        logger.info(f"Requesting full meme generation for: {request.topic}")
        response = await meme_gen_ref.ask(request, timeout=60000)
        
        logger.info(f"Generated meme ID: {response.get('id', 'N/A')}")
        logger.info(f"Meme text: {response.get('text', 'N/A')}")
        logger.info(f"Image URL: {response.get('image_url', 'N/A')[:100]}...")
        logger.info(f"Final score: {response.get('score', 0):.2f}")
        logger.info(f"Refinements: {response.get('refinement_count', 0)}")
        logger.info(f"Total time: {response.get('generation_time', 0):.2f}s")
        
    except Exception as e:
        logger.error(f"Full meme generation test failed: {e}")


async def show_metrics(meme_gen_ref, text_gen_ref, image_gen_ref):
    """Display metrics from all actors."""
    logger.info("=== Actor Metrics ===")
    
    try:
        # Get metrics from all actors
        metrics_request = GetMetricsRequest()
        
        meme_metrics = await meme_gen_ref.ask(metrics_request)
        text_metrics = await text_gen_ref.ask(metrics_request)
        image_metrics = await image_gen_ref.ask(metrics_request)
        
        logger.info("Meme Generator Metrics:")
        for key, value in meme_metrics.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("Text Generator Metrics:")
        for key, value in text_metrics.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("Image Generator Metrics:")
        for key, value in image_metrics.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling and circuit breaker functionality."""
    logger.info("=== Testing Error Handling ===")
    
    system = ActorSystem("error_test_system")
    await system.start()
    
    try:
        # Create actors
        text_gen_ref = await system.spawn(TextGeneratorActor, "error_test_text")
        
        # Test with invalid request that should trigger error handling
        invalid_request = GenerateTextRequest(
            topic="",  # Empty topic might cause issues
            format="nonexistent_format",
            style=None
        )
        
        try:
            response = await text_gen_ref.ask(invalid_request, timeout=10000)
            logger.info(f"Unexpected success: {response}")
        except Exception as e:
            logger.info(f"Expected error occurred: {e}")
            
        # Check circuit breaker state
        metrics = await text_gen_ref.ask(GetMetricsRequest())
        logger.info(f"Circuit breaker state: {metrics.get('circuit_breaker_state', 'unknown')}")
        
    finally:
        await system.stop()


if __name__ == "__main__":
    """Run the demonstration."""
    
    logger.info("Starting Meme Generation Actor System Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run main demonstration
        asyncio.run(demonstrate_actor_system())
        
        # Run error handling demonstration
        asyncio.run(demonstrate_error_handling())
        
        logger.info("=" * 60)
        logger.info("Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise