"""Command-line interface for the DSPy Meme Generator."""

import os
import typer
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from dspy_meme_gen.agents.router import RouterAgent
from dspy_meme_gen.agents.prompt_generator import PromptGenerationAgent
from dspy_meme_gen.agents.image_renderer import ImageRenderingAgent

# Initialize Typer app
app = typer.Typer(help="DSPy-powered Meme Generator CLI")

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    os.getenv("LOG_FILE", "logs/dspy_meme_gen.log"),
    level=os.getenv("LOG_LEVEL", "INFO"),
    rotation="1 day"
)

@app.command()
def generate(
    topic: str = typer.Option(..., "--topic", "-t", help="Meme topic"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Meme format"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Visual style"),
    output: Path = typer.Option("./meme.jpg", "--output", "-o", help="Output path")
) -> None:
    """
    Generate a meme based on specified parameters.
    
    Args:
        topic: The main topic or subject of the meme
        format: Optional specific meme format to use
        style: Optional visual style for the meme
        output: Path to save the generated meme
    """
    try:
        # Build user request
        user_request = f"Create a meme about {topic}"
        if format:
            user_request += f" using the {format} format"
        if style:
            user_request += f" in {style} style"
            
        # Initialize agents
        router = RouterAgent()
        prompt_generator = PromptGenerationAgent()
        image_renderer = ImageRenderingAgent()
        
        # Get routing information
        route_result = router(user_request)
        
        # Generate prompts
        format_details = {"name": route_result["format"]} if route_result["format"] else {}
        prompt_result = prompt_generator(
            topic=route_result["topic"],
            format_details=format_details,
            constraints=route_result["constraints"]
        )
        
        # Generate image
        image_result = image_renderer(
            image_prompt=prompt_result["image_prompt"],
            caption=prompt_result["caption"],
            format_details=format_details
        )
        
        # Download and save the image
        import requests
        response = requests.get(image_result["image_url"])
        response.raise_for_status()
        
        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        with open(output, "wb") as f:
            f.write(response.content)
            
        typer.echo(f"Meme generated successfully and saved to {output}")
        typer.echo(f"Caption: {prompt_result['caption']}")
        typer.echo(f"Reasoning: {prompt_result['reasoning']}")
        
    except Exception as e:
        logger.error(f"Error generating meme: {str(e)}")
        typer.echo(f"Failed to generate meme: {str(e)}", err=True)
        raise typer.Exit(1)

def main() -> None:
    """Entry point for the CLI application."""
    app() 