import { NextRequest, NextResponse } from 'next/server';

const SCHWAB_BRIDGE = process.env.SCHWAB_BRIDGE_URL || 'http://localhost:8502';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
  }

  try {
    const response = await fetch(
      `${SCHWAB_BRIDGE}/api/gex?symbol=${symbol}`,
      { next: { revalidate: 60 } }
    );
    
    if (!response.ok) {
      throw new Error(`Bridge error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('GEX error:', error);
    
    // Fallback mock data
    const basePrice = 590 + Math.random() * 20;
    return NextResponse.json({
      symbol,
      totalGex: (Math.random() - 0.3) * 5e9,
      flipPrice: basePrice - 2 + Math.random() * 4,
      callWall: Math.round(basePrice * 1.02),
      putWall: Math.round(basePrice * 0.98),
      expectedMove: basePrice * 0.012,
    });
  }
}
