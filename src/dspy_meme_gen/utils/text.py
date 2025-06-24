"""Text processing utilities."""

from typing import List, Optional, Tuple


def format_meme_text(
    text: str, max_length: int = 50, max_lines: int = 3, capitalize: bool = True
) -> str:
    """Format text for meme display.

    Args:
        text: Input text
        max_length: Maximum characters per line
        max_lines: Maximum number of lines
        capitalize: Whether to capitalize text

    Returns:
        str: Formatted text

    Raises:
        ValueError: If text is too long for specified constraints
    """
    # Capitalize if requested
    if capitalize:
        text = text.upper()

    # Split into words
    words = text.split()

    # Build lines
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        # Check if adding this word exceeds max_length
        if current_length + len(word) + (1 if current_line else 0) <= max_length:
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)
        else:
            # Start new line
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    # Add last line
    if current_line:
        lines.append(" ".join(current_line))

    # Check constraints
    if len(lines) > max_lines:
        raise ValueError(f"Text requires {len(lines)} lines but max_lines is {max_lines}")

    return "\n".join(lines)


def validate_meme_text(
    text: str,
    min_length: int = 1,
    max_length: int = 200,
    disallowed_words: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate text for meme usage.

    Args:
        text: Input text
        min_length: Minimum text length
        max_length: Maximum text length
        disallowed_words: List of words to disallow

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Check length
    if len(text) < min_length:
        return False, f"Text length {len(text)} is below minimum {min_length}"
    if len(text) > max_length:
        return False, f"Text length {len(text)} exceeds maximum {max_length}"

    # Check disallowed words
    if disallowed_words:
        lower_text = text.lower()
        found_words = [word for word in disallowed_words if word.lower() in lower_text]
        if found_words:
            return False, f"Text contains disallowed words: {', '.join(found_words)}"

    return True, None


def generate_meme_text_variants(
    text: str, num_variants: int = 3, max_length: int = 50
) -> List[str]:
    """Generate variations of meme text.

    Args:
        text: Base text
        num_variants: Number of variants to generate
        max_length: Maximum length per variant

    Returns:
        List[str]: List of text variants
    """
    variants = [text]  # Original text is first variant

    # Simple transformations for demonstration
    # In practice, this could use more sophisticated NLP
    transformations = [
        lambda t: t.upper(),
        lambda t: t.lower(),
        lambda t: t.title(),
        lambda t: "!".join(t.split()),
        lambda t: t.replace(" ", "_"),
        lambda t: t[::-1],  # Reverse
    ]

    for transform in transformations:
        if len(variants) >= num_variants:
            break
        variant = transform(text)
        if len(variant) <= max_length and variant not in variants:
            variants.append(variant)

    return variants[:num_variants]


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.

    Args:
        text: Input text

    Returns:
        List[str]: List of hashtags (without #)
    """
    words = text.split()
    return [word[1:] for word in words if word.startswith("#")]


def generate_hashtags(
    text: str, num_tags: int = 3, min_length: int = 3, max_length: int = 20
) -> List[str]:
    """Generate relevant hashtags for meme text.

    Args:
        text: Input text
        num_tags: Number of hashtags to generate
        min_length: Minimum hashtag length
        max_length: Maximum hashtag length

    Returns:
        List[str]: List of generated hashtags
    """
    # Split into words and filter by length
    words = [word.strip(".,!?") for word in text.split() if min_length <= len(word) <= max_length]

    # Remove duplicates and sort by length (shorter first)
    unique_words = sorted(set(words), key=len)

    # Convert to hashtags
    hashtags = []
    for word in unique_words:
        if len(hashtags) >= num_tags:
            break
        # Remove special characters and spaces
        tag = "".join(c for c in word if c.isalnum())
        if tag and tag not in hashtags:
            hashtags.append(tag.lower())

    return hashtags
