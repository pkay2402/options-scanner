// Trading Hub Type Definitions

export interface Quote {
  symbol: string;
  lastPrice: number;
  netChange: number;
  netPercentChange: number;
  bidPrice: number;
  askPrice: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  close: number;
}

export interface WatchlistStock {
  symbol: string;
  price: number;
  change: number;
  changePct: number;
  volume: number;
  whaleScore: number;
  gexScore: number;
  compositeScore: number;
  signals: string[];
}

export interface WhaleFlow {
  symbol: string;
  type: 'CALL' | 'PUT';
  strike: number;
  expiry: string;
  volume: number;
  openInterest: number;
  volOiRatio: number;
  premium: number;
  whaleScore: number;
  timestamp: string;
}

export interface ScannerSignal {
  id: string;
  symbol: string;
  signalType: 'MACD' | 'TTM_SQUEEZE' | 'VPB' | 'CYCLE' | 'WHALE' | 'ZSCORE';
  direction: 'bullish' | 'bearish' | 'neutral';
  description: string;
  price: number;
  timestamp: string;
}

export interface VolumeWall {
  strike: number;
  callVolume: number;
  putVolume: number;
  callOI: number;
  putOI: number;
  netGamma: number;
}

export interface GEXData {
  symbol: string;
  totalGex: number;
  flipPrice: number;
  callWall: number;
  putWall: number;
  expectedMove: number;
}

export interface MarketPulse {
  spy: Quote | null;
  qqq: Quote | null;
  vix: Quote | null;
  es: Quote | null;
  gexSummary: {
    spyGex: number;
    qqqGex: number;
    regime: 'positive' | 'negative';
  };
}

export interface OptionsChain {
  symbol: string;
  underlyingPrice: number;
  calls: OptionContract[];
  puts: OptionContract[];
}

export interface OptionContract {
  symbol: string;
  strike: number;
  expiry: string;
  type: 'CALL' | 'PUT';
  bid: number;
  ask: number;
  mark: number;
  volume: number;
  openInterest: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  iv: number;
  dte: number;
}

export interface PriceCandle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

// API Response types
export interface APIResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: string;
}
