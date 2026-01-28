// Schwab API integration for Next.js
// This proxies requests to your Python backend or directly to Schwab

const SCHWAB_API_BASE = 'https://api.schwabapi.com';
const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL || 'http://localhost:8501';

interface TokenData {
  access_token: string;
  refresh_token: string;
  expires_at: number;
}

let cachedToken: TokenData | null = null;

export async function getSchwabToken(): Promise<string | null> {
  // For now, we'll proxy through the Python backend which handles token management
  // In production, you'd implement full OAuth2 flow here
  
  try {
    // Try to read token from the schwab_client.json file via Python backend
    const response = await fetch(`${PYTHON_BACKEND}/api/schwab/token`, {
      method: 'GET',
    });
    
    if (response.ok) {
      const data = await response.json();
      return data.access_token;
    }
  } catch (error) {
    console.error('Error getting Schwab token:', error);
  }
  
  return null;
}

export async function schwabFetch(endpoint: string, options: RequestInit = {}) {
  const token = await getSchwabToken();
  
  if (!token) {
    throw new Error('No valid Schwab token available');
  }

  const response = await fetch(`${SCHWAB_API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`Schwab API error: ${response.status}`);
  }

  return response.json();
}

// Helper to format quote response
export function formatQuoteResponse(data: Record<string, any>) {
  const quote = data?.quote || data;
  return {
    symbol: quote.symbol || data.symbol,
    lastPrice: quote.lastPrice || quote.mark || 0,
    netChange: quote.netChange || 0,
    netPercentChange: quote.netPercentChangeInDouble || quote.netPercentChange || 0,
    bidPrice: quote.bidPrice || 0,
    askPrice: quote.askPrice || 0,
    volume: quote.totalVolume || quote.volume || 0,
    high: quote.highPrice || quote['52WkHigh'] || 0,
    low: quote.lowPrice || quote['52WkLow'] || 0,
    open: quote.openPrice || 0,
    close: quote.closePrice || quote.lastPrice || 0,
  };
}
