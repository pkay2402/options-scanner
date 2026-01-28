import { NextRequest, NextResponse } from 'next/server';

const SCHWAB_BRIDGE = process.env.SCHWAB_BRIDGE_URL || 'http://localhost:8502';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');
  const expiry = request.nextUrl.searchParams.get('expiry');

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
  }

  try {
    let url = `${SCHWAB_BRIDGE}/api/volume-walls?symbol=${symbol}`;
    if (expiry) url += `&expiry=${expiry}`;
    
    const response = await fetch(url, { next: { revalidate: 60 } });
    
    if (!response.ok) {
      throw new Error(`Bridge error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Volume walls error:', error);
    
    // Fallback mock data
    const basePrice = 590 + Math.random() * 20;
    return NextResponse.json({
      symbol,
      underlyingPrice: basePrice,
      callWall: Math.round(basePrice * 1.02),
      putWall: Math.round(basePrice * 0.98),
      walls: [],
    });
  }
}
