"""Image processing utilities."""

from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


def resize_image(
    image: Union[Image.Image, BytesIO, Path, str],
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
) -> Image.Image:
    """Resize an image to target dimensions.

    Args:
        image: Input image as PIL Image, BytesIO, Path, or file path
        target_size: Desired (width, height)
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        PIL.Image.Image: Resized image

    Raises:
        ValueError: If image format is invalid
    """
    if isinstance(image, (BytesIO, Path, str)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Invalid image format")

    if maintain_aspect:
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        return img
    else:
        return img.resize(target_size, Image.Resampling.LANCZOS)


def add_text_overlay(
    image: Union[Image.Image, BytesIO, Path, str],
    text: str,
    position: str = "bottom",
    font_size: int = 40,
    font_path: Optional[str] = None,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 2,
) -> Image.Image:
    """Add text overlay to an image.

    Args:
        image: Input image
        text: Text to overlay
        position: Text position ('top', 'bottom', 'center')
        font_size: Font size in pixels
        font_path: Path to font file (optional)
        text_color: RGB color tuple for text
        outline_color: RGB color tuple for text outline
        outline_width: Width of text outline in pixels

    Returns:
        PIL.Image.Image: Image with text overlay

    Raises:
        ValueError: If image format or position is invalid
    """
    if isinstance(image, (BytesIO, Path, str)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Invalid image format")

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_width, text_height = draw.textsize(text, font=font)
    image_width, image_height = img.size

    if position == "top":
        text_position = ((image_width - text_width) // 2, 20)
    elif position == "bottom":
        text_position = ((image_width - text_width) // 2, image_height - text_height - 20)
    elif position == "center":
        text_position = ((image_width - text_width) // 2, (image_height - text_height) // 2)
    else:
        raise ValueError("Invalid position. Must be 'top', 'bottom', or 'center'")

    # Draw text outline
    for offset_x, offset_y in [
        (-outline_width, -outline_width),
        (-outline_width, outline_width),
        (outline_width, -outline_width),
        (outline_width, outline_width),
    ]:
        draw.text(
            (text_position[0] + offset_x, text_position[1] + offset_y),
            text,
            font=font,
            fill=outline_color,
        )

    # Draw main text
    draw.text(text_position, text, font=font, fill=text_color)

    return img


def convert_format(
    image: Union[Image.Image, BytesIO, Path, str], output_format: str = "JPEG", quality: int = 95
) -> BytesIO:
    """Convert image format.

    Args:
        image: Input image
        output_format: Desired output format ('JPEG', 'PNG', etc.)
        quality: Output quality (1-100, JPEG only)

    Returns:
        BytesIO: Converted image data

    Raises:
        ValueError: If image format is invalid
    """
    if isinstance(image, (BytesIO, Path, str)):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Invalid image format")

    output = BytesIO()
    img.save(output, format=output_format, quality=quality)
    output.seek(0)
    return output


def create_meme_template(
    background: Union[Image.Image, BytesIO, Path, str],
    text_positions: list[str],
    output_size: Tuple[int, int] = (800, 600),
) -> Image.Image:
    """Create a meme template from a background image.

    Args:
        background: Background image
        text_positions: List of text positions ('top', 'bottom', 'center')
        output_size: Desired output size

    Returns:
        PIL.Image.Image: Prepared meme template

    Raises:
        ValueError: If image format is invalid or text positions are invalid
    """
    # Validate text positions
    valid_positions = {"top", "bottom", "center"}
    if not all(pos in valid_positions for pos in text_positions):
        raise ValueError(f"Invalid text position. Must be one of {valid_positions}")

    # Load and resize background
    template = resize_image(background, output_size)

    # Add placeholder text to mark positions
    for position in text_positions:
        template = add_text_overlay(
            template,
            f"[{position.upper()} TEXT]",
            position=position,
            font_size=30,
            text_color=(128, 128, 128),
            outline_color=(0, 0, 0),
            outline_width=1,
        )

    return template
