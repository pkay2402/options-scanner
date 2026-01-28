import { NextResponse } from 'next/server';

const SCHWAB_BRIDGE = process.env.SCHWAB_BRIDGE_URL || 'http://localhost:8502';

export async function GET() {
  try {
    const response = await fetch(`${SCHWAB_BRIDGE}/api/market-pulse`, {
      next: { revalidate: 15 } // Cache for 15 seconds
    });
    
    if (!response.ok) {
      throw new Error(`Bridge error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Market pulse error:', error);
    
    // Fallback to mock data if bridge is down
    return NextResponse.json({
      spy: {
        symbol: 'SPY',
        lastPrice: 594.25,
        netChange: 2.15,
        netPercentChange: 0.36,
        volume: 45_000_000,
      },
      qqq: {
        symbol: 'QQQ',
        lastPrice: 518.75,
        netChange: 3.20,
        netPercentChange: 0.62,
        volume: 32_000_000,
      },
      vix: {
        symbol: 'VIX',
        lastPrice: 14.25,
        netChange: -0.50,
        netPercentChange: -3.39,
      },
    });
  }
}
