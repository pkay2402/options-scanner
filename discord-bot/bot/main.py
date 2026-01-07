#!/usr/bin/env python3
"""
Discord Options Trading Bot - Main Entry Point
Uses service account authentication (Option A) with Schwab API
"""

import discord
from discord import app_commands
from discord.ext import commands
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to access existing src/ code
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.schwab_client import SchwabClient

# Import bot services using relative path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bot.services.schwab_service import SchwabService
from bot.services.multi_channel_alert_service import MultiChannelAlertService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from discord-bot/.env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class OptionsTradingBot(commands.Bot):
    """
    Discord bot for options trading analysis
    Uses shared Schwab service account (Option A authentication)
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix='/',
            intents=intents,
            description='Options Trading Analysis Bot'
        )
        
        # Initialize Schwab service (handles auth and token refresh)
        self.schwab_service = None
        
        # Initialize multi-channel automated alert service
        self.multi_alert_service = None
        
    async def setup_hook(self):
        """Load commands and initialize services"""
        try:
            # Initialize Schwab service with auto-refresh
            logger.info("Initializing Schwab API service...")
            self.schwab_service = SchwabService()
            await self.schwab_service.start()
            logger.info("Schwab service initialized successfully")
            
            # Initialize multi-channel alert service
            logger.info("Initializing multi-channel alert service...")
            self.multi_alert_service = MultiChannelAlertService(self)
            logger.info("Multi-channel alert service initialized")
            
            # Load command modules
            logger.info("Loading command modules...")
            await self.load_extension('bot.commands.dte_commands')
            await self.load_extension('bot.commands.gamma_map')
            await self.load_extension('bot.commands.whale_score')
            await self.load_extension('bot.commands.ema_cloud')
            await self.load_extension('bot.commands.alerts')
            await self.load_extension('bot.commands.watchlist_scanner')
            await self.load_extension('bot.commands.zscore_scanner')
            await self.load_extension('bot.commands.opening_move')
            logger.info("Command modules loaded")
            
            # Sync slash commands with Discord
            logger.info("Syncing slash commands...")
            # Global sync (takes up to 1 hour to propagate)
            await self.tree.sync()
            logger.info("Slash commands synced globally")
            # Note: Guild-specific sync happens in on_ready() after bot connects
            
        except Exception as e:
            logger.error(f"Error in setup_hook: {e}", exc_info=True)
            raise

    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'✅ Bot is online as {self.user}')
        logger.info(f'📊 Connected to {len(self.guilds)} server(s)')
        logger.info(f'🤖 Bot ID: {self.user.id}')
        
        # Sync commands to each guild for instant availability
        # (Global sync in setup_hook takes up to 1 hour)
        for guild in self.guilds:
            try:
                await self.tree.sync(guild=guild)
                logger.info(f"⚡ Commands synced to guild: {guild.name} (ID: {guild.id})")
            except Exception as e:
                logger.error(f"Failed to sync commands to guild {guild.id}: {e}")
        
        # Auto-start multi-channel alerts if configured
        if self.multi_alert_service:
            await self.multi_alert_service.auto_start_if_configured()        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="options flow 📊"
            )
        )
        
    async def on_command_error(self, ctx, error):
        """Global error handler"""
        if isinstance(error, commands.CommandNotFound):
            return  # Ignore unknown commands
        
        logger.error(f"Command error: {error}", exc_info=True)
        await ctx.send(f"❌ An error occurred: {str(error)}")
        
    async def close(self):
        """Cleanup when bot shuts down"""
        logger.info("Shutting down bot...")
        if self.multi_alert_service:
            await self.multi_alert_service.stop()
        if self.schwab_service:
            await self.schwab_service.stop()
        await super().close()


def main():
    """Main entry point"""
    # Get Discord bot token
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.error("DISCORD_BOT_TOKEN not found in environment variables")
        sys.exit(1)
    
    # Verify Schwab credentials
    if not os.getenv('SCHWAB_CLIENT_ID') or not os.getenv('SCHWAB_CLIENT_SECRET'):
        logger.error("Schwab credentials not found in environment variables")
        sys.exit(1)
    
    # Create and run bot
    bot = OptionsTradingBot()
    
    try:
        logger.info("Starting Discord bot...")
        bot.run(token, log_handler=None)  # We use our own logging
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
