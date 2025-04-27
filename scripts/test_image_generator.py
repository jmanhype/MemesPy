#!/usr/bin/env python
"""Test script for the ImageGenerator module.

This script demonstrates how to use the ImageGenerator with different provider backends,
including the placeholder implementation for GPT-4o.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the ImageGenerator
from src.dspy_meme_gen.dspy_modules.image_generator import ImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Rich console
console = Console()

def main():
    """Run the test script."""
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[bold red]ERROR:[/bold red] OPENAI_API_KEY environment variable not set.\n"
                "Please set it in your .env file or environment variables.",
                title="Missing API Key",
                border_style="red",
            )
        )
        return
    
    # Create a table to display the results
    table = Table(title="Image Generation Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Prompt", style="green")
    table.add_column("Result", style="yellow")
    
    # Test cases
    test_cases = [
        {
            "provider": "placeholder",
            "prompt": "Python programming meme with a snake wearing glasses coding",
            "style": None,
            "meme_text": None,
        },
        {
            "provider": "dalle",
            "prompt": "Python programming meme with a snake wearing glasses coding",
            "style": None,
            "meme_text": None,
        },
        {
            "provider": "gpt4o",
            "prompt": "Python programming meme with a snake wearing glasses coding",
            "style": "comic book",
            "meme_text": "When your code works on the first try",
        },
    ]
    
    # Run the tests
    for test in test_cases:
        console.print(f"Testing {test['provider']} provider...", style="bold blue")
        
        # Create the generator with the specified provider
        generator = ImageGenerator(provider=test["provider"])
        
        try:
            # Generate the image
            result = generator.generate(
                prompt=test["prompt"],
                style=test["style"],
                meme_text=test["meme_text"],
            )
            
            # Add the result to the table
            table.add_row(
                test["provider"],
                test["prompt"][:50] + "..." if len(test["prompt"]) > 50 else test["prompt"],
                result[:50] + "..." if len(result) > 50 else result,
            )
            
            console.print(f"✅ {test['provider']} successful", style="green")
        except Exception as e:
            # Add the error to the table
            table.add_row(
                test["provider"],
                test["prompt"][:50] + "..." if len(test["prompt"]) > 50 else test["prompt"],
                f"[bold red]ERROR: {str(e)}[/bold red]",
            )
            
            console.print(f"❌ {test['provider']} failed: {str(e)}", style="red")
    
    # Print the table
    console.print("\n")
    console.print(table)
    
    # Print additional information about GPT-4o
    console.print(
        Panel(
            "GPT-4o image generation is not yet available through the API.\n"
            "This script demonstrates the usage of the ImageGenerator with different providers,\n"
            "including a placeholder implementation for GPT-4o that will be updated when the API is released.",
            title="GPT-4o Image Generation",
            border_style="yellow",
        )
    )

if __name__ == "__main__":
    main() 