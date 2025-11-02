"""
Discord webhook alerts for options trading platform
"""

import requests
import logging
from typing import Optional
from datetime import datetime
from .config import get_discord_webhook

logger = logging.getLogger(__name__)


def send_discord_alert(
    title: str,
    message: str,
    color: Optional[int] = None,
    fields: Optional[dict] = None
) -> bool:
    """
    Send an alert to Discord webhook
    
    Args:
        title: Alert title
        message: Alert message/description
        color: Embed color (decimal) - Green: 5763719, Red: 15548997, Blue: 3447003
        fields: Dictionary of field name -> field value for structured data
    
    Returns:
        True if sent successfully, False otherwise
    """
    webhook_url = get_discord_webhook()
    
    if not webhook_url:
        logger.warning("Discord webhook not configured")
        return False
    
    # Default to blue if no color specified
    if color is None:
        color = 3447003  # Blue
    
    # Build embed
    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": "Options Trading Platform"
        }
    }
    
    # Add fields if provided
    if fields:
        embed["fields"] = [
            {"name": name, "value": str(value), "inline": True}
            for name, value in fields.items()
        ]
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Discord alert sent: {title}")
        return True
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")
        return False


def send_trade_alert(
    symbol: str,
    strike: float,
    expiry: str,
    option_type: str,
    premium: float,
    volume: int,
    action: str = "BUY"
) -> bool:
    """
    Send a formatted trade alert to Discord
    
    Args:
        symbol: Stock symbol
        strike: Strike price
        expiry: Expiration date
        option_type: 'CALL' or 'PUT'
        premium: Premium paid/received
        volume: Contract volume
        action: 'BUY' or 'SELL'
    
    Returns:
        True if sent successfully
    """
    color = 5763719 if action == "BUY" else 15548997  # Green for BUY, Red for SELL
    
    title = f"🎯 {action} {option_type} Alert: {symbol}"
    message = f"Large options trade detected"
    
    fields = {
        "Strike": f"${strike:.2f}",
        "Expiry": expiry,
        "Type": option_type,
        "Premium": f"${premium:,.0f}",
        "Volume": f"{volume:,} contracts",
        "Action": action
    }
    
    return send_discord_alert(title, message, color, fields)


def send_signal_alert(
    signal_type: str,
    symbol: str,
    description: str,
    strength: str = "MEDIUM"
) -> bool:
    """
    Send a market signal alert to Discord
    
    Args:
        signal_type: Type of signal (e.g., "Gamma Squeeze", "Unusual Flow")
        symbol: Stock symbol
        description: Signal description
        strength: Signal strength ('LOW', 'MEDIUM', 'HIGH')
    
    Returns:
        True if sent successfully
    """
    # Color based on strength
    colors = {
        "LOW": 3447003,    # Blue
        "MEDIUM": 16776960, # Yellow
        "HIGH": 15548997    # Red
    }
    color = colors.get(strength, 3447003)
    
    title = f"📊 {signal_type}: {symbol}"
    
    fields = {
        "Signal": signal_type,
        "Strength": strength,
        "Symbol": symbol
    }
    
    return send_discord_alert(title, description, color, fields)


def send_gamma_alert(
    symbol: str,
    strike: float,
    gamma_exposure: float,
    message: str
) -> bool:
    """
    Send a gamma-related alert to Discord
    
    Args:
        symbol: Stock symbol
        strike: Strike price with high gamma
        gamma_exposure: Gamma exposure value
        message: Alert message
    
    Returns:
        True if sent successfully
    """
    title = f"⚡ High Gamma Detected: {symbol}"
    
    fields = {
        "Symbol": symbol,
        "Strike": f"${strike:.2f}",
        "Gamma Exposure": f"${gamma_exposure:,.0f}"
    }
    
    return send_discord_alert(title, message, 16776960, fields)  # Yellow
