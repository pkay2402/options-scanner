"""
Schwab API Service for Discord Bot
Handles authentication, token refresh, and provides Schwab client access
"""

import asyncio
import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Access existing Schwab client
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.api.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


class SchwabService:
    """
    Manages Schwab API client with automatic token refresh
    Uses service account authentication (shared across all Discord users)
    """
    
    def __init__(self):
        self.client = None
        self._refresh_task = None
        self._is_running = False
        
    async def start(self):
        """Initialize Schwab client and start token refresh loop"""
        try:
            # Initialize Schwab client (uses existing schwab_client.json)
            logger.info("Authenticating with Schwab API...")
            self.client = SchwabClient()
            
            # Authenticate using stored tokens or setup flow
            if not self.client.authenticate():
                raise Exception("Failed to authenticate with Schwab API. Check credentials.")
            
            logger.info("✅ Schwab authentication successful")
            
            # Start background token refresh task
            self._is_running = True
            self._refresh_task = asyncio.create_task(self._token_refresh_loop())
            logger.info("Token refresh loop started")
            
        except Exception as e:
            logger.error(f"Failed to start Schwab service: {e}", exc_info=True)
            raise
            
    async def _token_refresh_loop(self):
        """Background task to refresh Schwab token periodically"""
        logger.info("Token refresh loop initiated (checks every 25 minutes)")
        
        while self._is_running:
            try:
                # Wait 25 minutes (tokens expire after 30 minutes)
                await asyncio.sleep(25 * 60)
                
                if not self._is_running:
                    break
                    
                logger.info("Refreshing Schwab access token...")
                
                # Ensure session is valid (will refresh if needed)
                if self.client.ensure_valid_session():
                    logger.info("✅ Token refreshed successfully")
                else:
                    logger.error("❌ Token refresh failed - re-authenticating...")
                    if not self.client.authenticate():
                        logger.critical("Failed to re-authenticate with Schwab API!")
                        
            except asyncio.CancelledError:
                logger.info("Token refresh loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in token refresh loop: {e}", exc_info=True)
                # Continue loop even on error
                await asyncio.sleep(60)  # Wait 1 minute before retry
                
    async def stop(self):
        """Stop the service and cleanup"""
        logger.info("Stopping Schwab service...")
        self._is_running = False
        
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Schwab service stopped")
        
    def get_client(self) -> SchwabClient:
        """
        Get the authenticated Schwab client
        
        Returns:
            SchwabClient: Authenticated Schwab API client
            
        Raises:
            RuntimeError: If service not started or client not initialized
        """
        if not self.client:
            raise RuntimeError("Schwab service not started. Call start() first.")
        return self.client
        
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self.client is not None and self.client.check_session()
