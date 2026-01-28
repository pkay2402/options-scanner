import { NextRequest, NextResponse } from 'next/server';

const SCHWAB_BRIDGE = process.env.SCHWAB_BRIDGE_URL || 'http://localhost:8502';

export async function GET(request: NextRequest) {
  const symbols = request.nextUrl.searchParams.get('symbols') || 'SPY,QQQ,NVDA,TSLA,AAPL,AMZN,META,AMD';
  const limit = request.nextUrl.searchParams.get('limit') || '20';

  try {
    const response = await fetch(
      `${SCHWAB_BRIDGE}/api/whale-flows?symbols=${symbols}&limit=${limit}`,
      { next: { revalidate: 30 } }
    );
    
    if (!response.ok) {
      throw new Error(`Bridge error: ${response.status}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Whale flows error:', error);
    
    // Return empty array on error
    return NextResponse.json({ data: [] });
  }
}
