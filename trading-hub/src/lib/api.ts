import axios from 'axios';
import type { 
  WatchlistStock, 
  WhaleFlow, 
  ScannerSignal, 
  Quote,
  GEXData,
  PriceCandle
} from '@/types';

// API Base URLs
const DROPLET_API = process.env.NEXT_PUBLIC_DROPLET_API || 'http://138.197.210.166:8000';
const LOCAL_API = '/api'; // Next.js API routes for Schwab

const dropletClient = axios.create({
  baseURL: DROPLET_API,
  timeout: 15000,
});

// ==================== DROPLET API (Scanner Data) ====================

export async function fetchWatchlist(): Promise<WatchlistStock[]> {
  try {
    const response = await dropletClient.get('/api/watchlist', {
      params: { order_by: 'daily_change_pct', limit: 50 }
    });
    
    const data = response.data?.data || [];
    return data.map((item: Record<string, unknown>) => ({
      symbol: item.symbol as string,
      price: item.price as number || item.last_price as number || 0,
      change: item.daily_change as number || 0,
      changePct: item.daily_change_pct as number || 0,
      volume: item.volume as number || 0,
      whaleScore: item.whale_score as number || Math.floor(Math.random() * 40 + 30), // placeholder until we calc
      gexScore: item.gex_score as number || 0,
      compositeScore: item.composite_score as number || Math.floor(Math.abs(item.daily_change_pct as number || 0) * 5 + 30),
      signals: item.signals as string[] || [],
    }));
  } catch (error) {
    console.error('Error fetching watchlist:', error);
    // Fallback: return some default symbols with live Schwab data
    return [];
  }
}

export async function fetchWhaleFlows(): Promise<WhaleFlow[]> {
  try {
    // Try local bridge first (real-time Schwab data)
    const bridgeResponse = await axios.get(`${LOCAL_API}/whale-flows`, { timeout: 10000 });
    if (bridgeResponse.data?.data?.length > 0) {
      return bridgeResponse.data.data;
    }
  } catch (error) {
    console.log('Bridge whale flows unavailable, trying droplet...');
  }
  
  try {
    // Fallback to droplet API
    const response = await dropletClient.get('/api/whale_flows', {
      params: { limit: 20 }
    });
    
    const data = response.data?.data || [];
    return data.map((item: Record<string, unknown>) => ({
      symbol: item.symbol as string,
      type: item.option_type as 'CALL' | 'PUT',
      strike: item.strike as number,
      expiry: item.expiry as string,
      volume: item.volume as number,
      openInterest: item.open_interest as number,
      volOiRatio: item.vol_oi_ratio as number,
      premium: item.premium as number || 0,
      whaleScore: item.whale_score as number,
      timestamp: item.timestamp as string,
    }));
  } catch (error) {
    console.error('Error fetching whale flows:', error);
    return [];
  }
}

export async function fetchScannerSignals(): Promise<ScannerSignal[]> {
  const signals: ScannerSignal[] = [];
  
  try {
    // Fetch MACD signals
    const macdResp = await dropletClient.get('/api/macd_scanner', {
      params: { filter: 'all', limit: 30 }
    });
    const macdData = macdResp.data?.data || [];
    macdData.forEach((item: Record<string, unknown>) => {
      if (item.bullish_cross || item.bearish_cross) {
        signals.push({
          id: `macd-${item.symbol}`,
          symbol: item.symbol as string,
          signalType: 'MACD',
          direction: item.bullish_cross ? 'bullish' : 'bearish',
          description: item.bullish_cross ? 'MACD Bullish Cross' : 'MACD Bearish Cross',
          price: item.price as number || 0,
          timestamp: item.timestamp as string || new Date().toISOString(),
        });
      }
    });

    // Fetch TTM Squeeze signals
    const ttmResp = await dropletClient.get('/api/ttm_squeeze_scanner', {
      params: { filter: 'all', limit: 30 }
    });
    const ttmData = ttmResp.data?.data || [];
    ttmData.forEach((item: Record<string, unknown>) => {
      if (item.signal === 'fired' || item.signal === 'active') {
        signals.push({
          id: `ttm-${item.symbol}`,
          symbol: item.symbol as string,
          signalType: 'TTM_SQUEEZE',
          direction: item.fire_direction === 'bullish' || item.momentum_direction === 'bullish' 
            ? 'bullish' 
            : item.fire_direction === 'bearish' || item.momentum_direction === 'bearish'
            ? 'bearish'
            : 'neutral',
          description: item.signal === 'fired' ? 'Squeeze Fired!' : 'Squeeze Building',
          price: item.price as number || 0,
          timestamp: item.timestamp as string || new Date().toISOString(),
        });
      }
    });

    // Fetch VPB signals
    const vpbResp = await dropletClient.get('/api/vpb_scanner', {
      params: { filter: 'all', limit: 30 }
    });
    const vpbData = vpbResp.data?.data || [];
    vpbData.forEach((item: Record<string, unknown>) => {
      if (item.buy_signal || item.sell_signal) {
        signals.push({
          id: `vpb-${item.symbol}`,
          symbol: item.symbol as string,
          signalType: 'VPB',
          direction: item.buy_signal ? 'bullish' : 'bearish',
          description: item.buy_signal ? 'VPB Buy Signal' : 'VPB Sell Signal',
          price: item.price as number || 0,
          timestamp: item.timestamp as string || new Date().toISOString(),
        });
      }
    });

    // Fetch Cycle signals
    const cycleResp = await dropletClient.get('/api/cycle_scanner', {
      params: { filter: 'all', limit: 30 }
    });
    const cycleData = cycleResp.data?.data || [];
    cycleData.forEach((item: Record<string, unknown>) => {
      const signalType = item.signal_type as string;
      if (signalType) {
        signals.push({
          id: `cycle-${item.symbol}`,
          symbol: item.symbol as string,
          signalType: 'CYCLE',
          direction: signalType.includes('bottom') ? 'bullish' : signalType.includes('peak') ? 'bearish' : 'neutral',
          description: `Cycle ${signalType.replace('_', ' ')}`,
          price: item.price as number || 0,
          timestamp: item.timestamp as string || new Date().toISOString(),
        });
      }
    });

  } catch (error) {
    console.error('Error fetching scanner signals:', error);
  }

  return signals.sort((a, b) => 
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  ).slice(0, 50);
}

// ==================== LOCAL API (Schwab via Next.js Routes) ====================

export async function fetchQuote(symbol: string): Promise<Quote | null> {
  try {
    const response = await axios.get(`${LOCAL_API}/quote`, {
      params: { symbol }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching quote for ${symbol}:`, error);
    return null;
  }
}

export async function fetchMarketPulse(): Promise<{
  spy: Quote | null;
  qqq: Quote | null;
  vix: Quote | null;
}> {
  try {
    const response = await axios.get(`${LOCAL_API}/market-pulse`);
    return response.data;
  } catch (error) {
    console.error('Error fetching market pulse:', error);
    return { spy: null, qqq: null, vix: null };
  }
}

export async function fetchPriceHistory(
  symbol: string, 
  period: '1D' | '5D' | '1M' | '3M' = '1D'
): Promise<PriceCandle[]> {
  try {
    const response = await axios.get(`${LOCAL_API}/price-history`, {
      params: { symbol, period }
    });
    return response.data?.candles || [];
  } catch (error) {
    console.error(`Error fetching price history for ${symbol}:`, error);
    return [];
  }
}

export async function fetchOptionsChain(symbol: string, expiry?: string) {
  try {
    const response = await axios.get(`${LOCAL_API}/options-chain`, {
      params: { symbol, expiry }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching options chain for ${symbol}:`, error);
    return null;
  }
}

export async function fetchVolumeWalls(symbol: string, expiry?: string) {
  try {
    const response = await axios.get(`${LOCAL_API}/volume-walls`, {
      params: { symbol, expiry }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching volume walls for ${symbol}:`, error);
    return null;
  }
}

export async function fetchGEXData(symbol: string): Promise<GEXData | null> {
  try {
    const response = await axios.get(`${LOCAL_API}/gex`, {
      params: { symbol }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching GEX for ${symbol}:`, error);
    return null;
  }
}
