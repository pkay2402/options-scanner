"""
Cached Schwab Client Utility
Provides a cached SchwabClient instance for Streamlit apps to reduce
authentication overhead and improve performance.
"""

import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_cached_client():
    """
    Get a cached SchwabClient instance.
    Uses @st.cache_resource to persist the client across reruns and sessions.
    
    Returns:
        SchwabClient: Authenticated client instance, or None if auth fails
    """
    try:
        from src.api.schwab_client import SchwabClient
        
        client = SchwabClient(interactive=False)
        
        if client.authenticate():
            logger.info("Cached SchwabClient authenticated successfully")
            return client
        else:
            logger.warning("Cached SchwabClient authentication failed")
            return None
            
    except Exception as e:
        logger.error(f"Error creating cached SchwabClient: {e}")
        return None


def get_client():
    """
    Get a SchwabClient - uses cached version if available.
    This is the main function pages should import and use.
    
    Returns:
        SchwabClient: Authenticated client instance
    """
    client = get_cached_client()
    
    if client is None:
        # Fallback: try to create a fresh client
        try:
            from src.api.schwab_client import SchwabClient
            client = SchwabClient(interactive=False)
            if not client.authenticate():
                logger.error("Fresh SchwabClient also failed to authenticate")
                return None
            return client
        except Exception as e:
            logger.error(f"Fallback client creation failed: {e}")
            return None
    
    return client


def clear_client_cache():
    """Clear the cached client - useful if token needs refresh"""
    get_cached_client.clear()
    logger.info("Cleared cached SchwabClient")
