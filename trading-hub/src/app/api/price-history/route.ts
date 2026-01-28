import { NextRequest, NextResponse } from 'next/server';

const SCHWAB_BRIDGE = process.env.SCHWAB_BRIDGE_URL || 'http://localhost:8502';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');
  const period = request.nextUrl.searchParams.get('period') || '1D';

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
  }

  try {
    const response = await fetch(
      `${SCHWAB_BRIDGE}/api/price-history?symbol=${symbol}&period=${period}`,
      { next: { revalidate: 30 } }
    );
    
    if (!response.ok) {
      throw new Error(`Bridge error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Price history error:', error);
    
    // Fallback to mock data
    const candles = generateMockCandles(period as '1D' | '5D' | '1M' | '3M');
    return NextResponse.json({ candles });
  }
}

function generateMockCandles(period: '1D' | '5D' | '1M' | '3M') {
  const now = Date.now();
  const candles = [];
  
  let numCandles: number;
  let intervalMs: number;
  
  switch (period) {
    case '1D': numCandles = 78; intervalMs = 5 * 60 * 1000; break;
    case '5D': numCandles = 5 * 78; intervalMs = 5 * 60 * 1000; break;
    case '1M': numCandles = 22; intervalMs = 24 * 60 * 60 * 1000; break;
    default: numCandles = 66; intervalMs = 24 * 60 * 60 * 1000;
  }

  let basePrice = 590 + Math.random() * 20;
  
  for (let i = 0; i < numCandles; i++) {
    const change = (Math.random() - 0.48) * 2;
    const open = basePrice;
    const close = basePrice + change;
    const high = Math.max(open, close) + Math.random() * 0.5;
    const low = Math.min(open, close) - Math.random() * 0.5;
    
    candles.push({
      time: now - (numCandles - i) * intervalMs,
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.floor(Math.random() * 1000000) + 100000,
    });
    
    basePrice = close;
  }

  return candles;
}
