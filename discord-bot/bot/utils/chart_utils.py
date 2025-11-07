"""
Chart utilities for Discord bot
Converts Plotly figures to Discord-compatible images
"""

import io
import logging
from typing import Optional
import discord

logger = logging.getLogger(__name__)


def plotly_to_discord_file(fig, filename: str, width: int = 1200, height: int = 800) -> Optional[discord.File]:
    """
    Convert Plotly figure to Discord file attachment
    
    Args:
        fig: Plotly figure object
        filename: Name for the file (should end in .png)
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        discord.File object or None if conversion fails
    """
    try:
        # Convert plotly figure to PNG bytes
        img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
        
        # Create Discord file from bytes
        file = discord.File(io.BytesIO(img_bytes), filename=filename)
        
        return file
        
    except Exception as e:
        logger.error(f"Error converting Plotly figure to Discord file: {e}")
        return None


def create_embed(
    title: str,
    description: str = None,
    color: discord.Color = discord.Color.blue(),
    fields: list = None,
    image_filename: str = None,
    footer_text: str = None
) -> discord.Embed:
    """
    Create a formatted Discord embed
    
    Args:
        title: Embed title
        description: Embed description
        color: Embed color
        fields: List of dicts with 'name', 'value', 'inline' keys
        image_filename: Attachment filename for image
        footer_text: Footer text
        
    Returns:
        discord.Embed object
    """
    embed = discord.Embed(
        title=title,
        description=description,
        color=color
    )
    
    # Add fields
    if fields:
        for field in fields:
            embed.add_field(
                name=field.get('name', ''),
                value=field.get('value', ''),
                inline=field.get('inline', False)
            )
    
    # Set image
    if image_filename:
        embed.set_image(url=f"attachment://{image_filename}")
    
    # Set footer
    if footer_text:
        embed.set_footer(text=footer_text)
    else:
        embed.set_footer(text="Data from Schwab API • Market hours only")
    
    return embed


def format_large_number(value: float) -> str:
    """
    Format large numbers for display (e.g., 1.5M, 2.3B)
    
    Args:
        value: Number to format
        
    Returns:
        Formatted string
    """
    abs_value = abs(value)
    
    if abs_value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """
    Format percentage value
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        
    Returns:
        Formatted string
    """
    return f"{value*100:.2f}%"


def truncate_text(text: str, max_length: int = 1024) -> str:
    """
    Truncate text to fit Discord field limits
    
    Args:
        text: Text to truncate
        max_length: Maximum length (Discord limit is 1024 for fields)
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
