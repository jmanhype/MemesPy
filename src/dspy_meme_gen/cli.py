"""Command-line interface for meme generation and management."""
import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .database.repository import MemeRepository
from .models.meme import MemeTemplate, GeneratedMeme, TrendingTopic
from .utils.config import get_config
from .utils.image import add_text_overlay, convert_format, resize_image
from .utils.text import format_meme_text, validate_meme_text, generate_hashtags

app = typer.Typer(
    name="dspy-meme-gen",
    help="A sophisticated meme generation pipeline using DSPy"
)
console = Console()

# Helper functions
def get_repository() -> MemeRepository:
    """Get repository instance."""
    config = get_config()
    return MemeRepository(config.database.url)

def display_template(template: MemeTemplate):
    """Display meme template details."""
    table = Table(title=f"Template: {template.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("ID", str(template.id))
    table.add_row("Description", template.description)
    table.add_row("Format Type", template.format_type)
    table.add_row("Example URL", template.example_url or "N/A")
    table.add_row("Popularity Score", f"{template.popularity_score:.2f}")
    table.add_row("Structure", str(template.structure))
    
    console.print(table)

def display_meme(meme: GeneratedMeme):
    """Display generated meme details."""
    table = Table(title=f"Meme: {meme.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Template ID", str(meme.template_id))
    table.add_row("Topic", meme.topic)
    table.add_row("Caption", meme.caption)
    table.add_row("Image URL", meme.image_url)
    table.add_row("Score", f"{meme.score:.2f}" if meme.score else "N/A")
    table.add_row("Created At", str(meme.created_at))
    
    console.print(table)

def display_trending(topic: TrendingTopic):
    """Display trending topic details."""
    table = Table(title=f"Trending Topic: {topic.topic}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("ID", str(topic.id))
    table.add_row("Source", topic.source)
    table.add_row("Relevance Score", f"{topic.relevance_score:.2f}")
    table.add_row("Timestamp", str(topic.timestamp))
    if topic.metadata:
        table.add_row("Metadata", str(topic.metadata))
    
    console.print(table)

# Template commands
@app.command()
def create_template(
    name: str = typer.Option(..., "--name", "-n", help="Template name"),
    description: str = typer.Option(..., "--desc", "-d", help="Template description"),
    format_type: str = typer.Option(..., "--type", "-t", help="Format type"),
    example_url: Optional[str] = typer.Option(None, "--url", "-u", help="Example URL"),
    structure: str = typer.Option(..., "--struct", "-s", help="Template structure (JSON)")
):
    """Create a new meme template."""
    try:
        repo = get_repository()
        template = asyncio.run(repo.create_meme_template(
            name=name,
            description=description,
            format_type=format_type,
            example_url=example_url,
            structure=structure
        ))
        display_template(template)
    except Exception as e:
        console.print(f"[red]Error creating template: {e}[/red]")

@app.command()
def get_template(
    template_id: int = typer.Argument(..., help="Template ID")
):
    """Get a meme template by ID."""
    try:
        repo = get_repository()
        template = asyncio.run(repo.get_meme_template(template_id))
        if template:
            display_template(template)
        else:
            console.print("[yellow]Template not found[/yellow]")
    except Exception as e:
        console.print(f"[red]Error getting template: {e}[/red]")

@app.command()
def list_templates(
    format_type: Optional[str] = typer.Option(None, "--type", "-t", help="Format type filter"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of templates to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of templates to skip")
):
    """List meme templates."""
    try:
        repo = get_repository()
        templates = asyncio.run(repo.list_meme_templates(
            format_type=format_type,
            limit=limit,
            offset=offset
        ))
        
        table = Table(title="Meme Templates")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Format Type", style="blue")
        table.add_column("Popularity", style="magenta")
        
        for template in templates:
            table.add_row(
                str(template.id),
                template.name,
                template.format_type,
                f"{template.popularity_score:.2f}"
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing templates: {e}[/red]")

# Meme commands
@app.command()
def generate_meme(
    template_id: int = typer.Option(..., "--template", "-t", help="Template ID"),
    topic: str = typer.Option(..., "--topic", "-p", help="Meme topic"),
    caption: str = typer.Option(..., "--caption", "-c", help="Meme caption"),
    image_prompt: Optional[str] = typer.Option(None, "--prompt", "-r", help="Image generation prompt")
):
    """Generate a new meme."""
    try:
        # Validate text
        is_valid, error = validate_meme_text(caption)
        if not is_valid:
            console.print(f"[red]Invalid caption: {error}[/red]")
            return
        
        # Format text
        formatted_caption = format_meme_text(caption)
        
        repo = get_repository()
        meme = asyncio.run(repo.create_generated_meme(
            template_id=template_id,
            topic=topic,
            caption=formatted_caption,
            image_prompt=image_prompt
        ))
        
        display_meme(meme)
        
        # Generate and display hashtags
        hashtags = generate_hashtags(topic + " " + caption)
        if hashtags:
            console.print("\nSuggested hashtags:")
            for tag in hashtags:
                console.print(f"#{tag}", style="blue")
    except Exception as e:
        console.print(f"[red]Error generating meme: {e}[/red]")

@app.command()
def get_meme(
    meme_id: int = typer.Argument(..., help="Meme ID")
):
    """Get a generated meme by ID."""
    try:
        repo = get_repository()
        meme = asyncio.run(repo.get_generated_meme(meme_id))
        if meme:
            display_meme(meme)
        else:
            console.print("[yellow]Meme not found[/yellow]")
    except Exception as e:
        console.print(f"[red]Error getting meme: {e}[/red]")

@app.command()
def list_memes(
    template_id: Optional[int] = typer.Option(None, "--template", "-t", help="Template ID filter"),
    topic: Optional[str] = typer.Option(None, "--topic", "-p", help="Topic filter"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of memes to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of memes to skip")
):
    """List generated memes."""
    try:
        repo = get_repository()
        memes = asyncio.run(repo.list_generated_memes(
            template_id=template_id,
            topic=topic,
            limit=limit,
            offset=offset
        ))
        
        table = Table(title="Generated Memes")
        table.add_column("ID", style="cyan")
        table.add_column("Template", style="green")
        table.add_column("Topic", style="blue")
        table.add_column("Score", style="magenta")
        
        for meme in memes:
            table.add_row(
                str(meme.id),
                str(meme.template_id),
                meme.topic,
                f"{meme.score:.2f}" if meme.score else "N/A"
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing memes: {e}[/red]")

# Trending commands
@app.command()
def list_trending(
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Source filter"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of topics to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of topics to skip")
):
    """List trending topics."""
    try:
        repo = get_repository()
        topics = asyncio.run(repo.list_trending_topics(
            source=source,
            limit=limit,
            offset=offset
        ))
        
        table = Table(title="Trending Topics")
        table.add_column("ID", style="cyan")
        table.add_column("Topic", style="green")
        table.add_column("Source", style="blue")
        table.add_column("Relevance", style="magenta")
        
        for topic in topics:
            table.add_row(
                str(topic.id),
                topic.topic,
                topic.source,
                f"{topic.relevance_score:.2f}"
            )
        
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing trending topics: {e}[/red]")

# Feedback commands
@app.command()
def add_feedback(
    meme_id: int = typer.Argument(..., help="Meme ID"),
    score: float = typer.Option(..., "--score", "-s", min=0, max=1, help="Feedback score (0-1)")
):
    """Add user feedback for a meme."""
    try:
        repo = get_repository()
        asyncio.run(repo.add_user_feedback(meme_id, score))
        console.print("[green]Feedback added successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error adding feedback: {e}[/red]")

def main():
    """Main entry point."""
    app() 