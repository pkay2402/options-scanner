"""
Schwab WebSocket Streaming Client
Handles real-time data streaming for Level 1 and Level 2 data
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
import threading
from queue import Queue

logger = logging.getLogger(__name__)


class SchwabStreamer:
    """
    WebSocket client for Schwab streaming API
    Supports Level 1 Options, Level 2 Options Book, and more
    """
    
    def __init__(self, access_token: str, streamer_info: Dict):
        """
        Initialize streaming client
        
        Args:
            access_token: OAuth access token
            streamer_info: Streamer info from account preferences
        """
        self.access_token = access_token
        self.streamer_info = streamer_info
        
        # WebSocket connection
        self.websocket = None
        self.is_connected = False
        
        # Message queue for incoming data
        self.message_queue = Queue()
        
        # Callbacks for different services
        self.callbacks = {
            'LEVELONE_OPTIONS': [],
            'OPTIONS_BOOK': [],
            'LEVELONE_EQUITIES': [],
            'CHART_EQUITY': []
        }
        
        # Background thread for async loop
        self.thread = None
        self.loop = None
        self.running = False
        
    def start(self):
        """Start streaming in background thread"""
        if self.running:
            logger.warning("Streamer already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        logger.info("Schwab streamer started")
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._close_connection(), self.loop)
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Schwab streamer stopped")
    
    def _run_event_loop(self):
        """Run async event loop in background thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self.loop.close()
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        url = self.streamer_info.get('streamerSocketUrl')
        
        if not url:
            logger.error("No streamer URL found")
            return
        
        try:
            async with websockets.connect(url) as websocket:
                self.websocket = websocket
                self.is_connected = True
                logger.info(f"Connected to Schwab WebSocket: {url}")
                
                # Send login request
                await self._login()
                
                # Listen for messages
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        self.is_connected = False
                        break
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
    
    async def _close_connection(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
    
    async def _login(self):
        """Send login request to Schwab streamer"""
        login_request = {
            "requests": [{
                "service": "ADMIN",
                "command": "LOGIN",
                "requestid": 0,
                "account": self.streamer_info.get('schwabClientAccountId'),
                "source": self.streamer_info.get('schwabClientCorrelId'),
                "parameters": {
                    "Authorization": self.access_token,
                    "SchwabClientChannel": self.streamer_info.get('schwabClientChannel'),
                    "SchwabClientFunctionId": self.streamer_info.get('schwabClientFunctionId')
                }
            }]
        }
        
        await self.websocket.send(json.dumps(login_request))
        logger.info("Login request sent")
        
        # Wait for login response
        response = await self.websocket.recv()
        logger.info(f"Login response: {response}")
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Route to appropriate callbacks based on service
            if 'data' in data:
                for item in data['data']:
                    service = item.get('service')
                    content = item.get('content', [])
                    
                    # Call registered callbacks
                    if service in self.callbacks:
                        for callback in self.callbacks[service]:
                            try:
                                callback(content)
                            except Exception as e:
                                logger.error(f"Callback error for {service}: {e}")
            
            # Also add to message queue for polling
            self.message_queue.put(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    def subscribe_level_one_options(self, symbols: List[str], callback: Optional[Callable] = None):
        """
        Subscribe to Level 1 options data (real-time quotes)
        
        Args:
            symbols: List of option symbols (e.g., ['AAPL_123124C500'])
            callback: Function to call when data arrives
        """
        if callback:
            self.callbacks['LEVELONE_OPTIONS'].append(callback)
        
        request = {
            "requests": [{
                "service": "LEVELONE_OPTIONS",
                "command": "SUBS",
                "requestid": self._get_request_id(),
                "account": self.streamer_info.get('schwabClientAccountId'),
                "source": self.streamer_info.get('schwabClientCorrelId'),
                "parameters": {
                    "keys": ",".join(symbols),
                    "fields": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41"
                }
            }]
        }
        
        asyncio.run_coroutine_threadsafe(
            self.websocket.send(json.dumps(request)), 
            self.loop
        )
        
        logger.info(f"Subscribed to Level 1 Options: {symbols}")
    
    def subscribe_options_book(self, symbols: List[str], callback: Optional[Callable] = None):
        """
        Subscribe to Level 2 options book (order book depth)
        
        Args:
            symbols: List of option symbols
            callback: Function to call when data arrives
        """
        if callback:
            self.callbacks['OPTIONS_BOOK'].append(callback)
        
        request = {
            "requests": [{
                "service": "OPTIONS_BOOK",
                "command": "SUBS",
                "requestid": self._get_request_id(),
                "account": self.streamer_info.get('schwabClientAccountId'),
                "source": self.streamer_info.get('schwabClientCorrelId'),
                "parameters": {
                    "keys": ",".join(symbols),
                    "fields": "0,1,2,3"
                }
            }]
        }
        
        asyncio.run_coroutine_threadsafe(
            self.websocket.send(json.dumps(request)), 
            self.loop
        )
        
        logger.info(f"Subscribed to Options Book: {symbols}")
    
    def unsubscribe(self, service: str, symbols: List[str]):
        """Unsubscribe from a service"""
        request = {
            "requests": [{
                "service": service,
                "command": "UNSUBS",
                "requestid": self._get_request_id(),
                "account": self.streamer_info.get('schwabClientAccountId'),
                "source": self.streamer_info.get('schwabClientCorrelId'),
                "parameters": {
                    "keys": ",".join(symbols)
                }
            }]
        }
        
        asyncio.run_coroutine_threadsafe(
            self.websocket.send(json.dumps(request)), 
            self.loop
        )
        
        logger.info(f"Unsubscribed from {service}: {symbols}")
    
    def get_messages(self, max_messages: int = 100) -> List[Dict]:
        """Poll message queue for new data"""
        messages = []
        while not self.message_queue.empty() and len(messages) < max_messages:
            messages.append(self.message_queue.get())
        return messages
    
    def _get_request_id(self) -> int:
        """Generate unique request ID"""
        import time
        return int(time.time() * 1000)
