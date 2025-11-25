"""
Integration helper for Schwab API with streaming
Handles authentication and streamer info retrieval
"""

import json
import os
from typing import Dict, Optional
from src.api.schwab_api import SchwabAPI


def get_schwab_client() -> Optional[SchwabAPI]:
    """
    Get authenticated Schwab API client
    Reads credentials from schwab_client.json
    """
    try:
        creds_path = os.path.join(os.path.dirname(__file__), '../../schwab_client.json')
        
        if not os.path.exists(creds_path):
            return None
        
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        
        client = SchwabAPI(
            client_id=creds.get('app_key'),
            client_secret=creds.get('app_secret'),
            redirect_uri=creds.get('redirect_uri', 'https://127.0.0.1:8182')
        )
        
        # Load tokens if available
        if 'access_token' in creds:
            client.access_token = creds['access_token']
        if 'refresh_token' in creds:
            client.refresh_token = creds['refresh_token']
        
        return client
        
    except Exception as e:
        print(f"Error loading Schwab credentials: {e}")
        return None


def get_streamer_info(client: SchwabAPI) -> Optional[Dict]:
    """
    Get streamer connection info from account preferences
    
    Returns:
        Dict with streamer info including:
        - streamerSocketUrl: WebSocket URL
        - schwabClientAccountId: Account ID
        - schwabClientCorrelId: Correlation ID
        - schwabClientChannel: Client channel
        - schwabClientFunctionId: Function ID
    """
    try:
        # Get account hash
        accounts = client.get_account_numbers()
        if not accounts:
            print("No accounts found")
            return None
        
        account_hash = accounts[0].get('hashValue')
        
        # Get user preferences which includes streamer info
        response = client.get(f'/trader/v1/userPreference')
        
        if response and 'streamerInfo' in response:
            streamer_info = response['streamerInfo'][0]
            return {
                'streamerSocketUrl': streamer_info.get('streamerSocketUrl'),
                'schwabClientAccountId': streamer_info.get('schwabClientAccountId'),
                'schwabClientCorrelId': streamer_info.get('schwabClientCorrelId'),
                'schwabClientChannel': streamer_info.get('schwabClientChannel'),
                'schwabClientFunctionId': streamer_info.get('schwabClientFunctionId')
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting streamer info: {e}")
        return None


def create_streamer_connection():
    """
    Convenience function to create a ready-to-use streamer
    
    Returns:
        Tuple of (SchwabStreamer, SchwabAPI) or (None, None)
    """
    from src.streaming.schwab_streamer import SchwabStreamer
    
    # Get authenticated client
    client = get_schwab_client()
    if not client:
        print("Failed to create Schwab client")
        return None, None
    
    # Get streamer info
    streamer_info = get_streamer_info(client)
    if not streamer_info:
        print("Failed to get streamer info")
        return None, None
    
    # Create streamer
    streamer = SchwabStreamer(
        access_token=client.access_token,
        streamer_info=streamer_info
    )
    
    return streamer, client


# Example usage in Streamlit pages:
"""
from src.streaming.integration import create_streamer_connection

# In your page
if 'streamer' not in st.session_state:
    streamer, client = create_streamer_connection()
    if streamer:
        st.session_state.streamer = streamer
        st.session_state.schwab_client = client
        
        # Start streaming
        streamer.start()
        
        # Subscribe to symbols
        streamer.subscribe_level_one_options(
            symbols=['SPY_241129C450'],
            callback=lambda data: handle_options_data(data)
        )
"""
