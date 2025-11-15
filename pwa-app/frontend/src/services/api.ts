export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export interface Quote {
  symbol: string;
  price?: number;
  mark?: number;
  change?: number;
  changePercent?: number;
  volume?: number;
  bid?: number;
  ask?: number;
  high?: number;
  low?: number;
  timestamp: string;
}

export interface VolumeWall {
  strike: number | null;
  volume: number;
  oi?: number;
  gex?: number;
}

export interface VolumeWallsData {
  symbol: string;
  underlying_price: number;
  expiry_date: string;
  call_wall: VolumeWall;
  put_wall: VolumeWall;
  net_call_wall?: { strike: number | null; volume: number };
  net_put_wall?: { strike: number | null; volume: number };
  flip_level: number | null;
  all_strikes: number[];
  call_volumes: Record<string, number>;
  put_volumes: Record<string, number>;
  call_oi?: Record<string, number>;
  put_oi?: Record<string, number>;
  net_volumes: Record<string, number>;
  gex_by_strike: Record<string, number>;
  totals: { call_vol: number; put_vol: number; net_vol: number; total_gex: number };
  timestamp: string;
}

export interface Candle {
  datetime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceHistory {
  symbol: string;
  candles: Candle[];
  timestamp: string;
}

export interface OptionFinderResultItem {
  strike: number;
  expiry: string;
  days_to_exp: number;
  option_type: 'Call' | 'Put';
  gamma: number;
  delta: number;
  vega: number;
  volume: number;
  open_interest: number;
  bid: number;
  ask: number;
  last: number;
  notional_gamma: number;
  signed_notional_gamma: number;
  moneyness: number;
  implied_volatility: number;
}

export interface OptionFinderResponse {
  symbol: string;
  underlying_price: number;
  expiries: string[];
  strikes: number[];
  heatmap: { x: string[]; y: number[]; z: number[][] };
  top_calls: OptionFinderResultItem[];
  top_puts: OptionFinderResultItem[];
  timestamp: string;
}

export interface NetPremiumHeatmapResponse {
  symbol: string;
  underlying_price: number;
  expiries: string[];
  strikes: number[];
  heatmap: { x: string[]; y: number[]; z: number[][] };
  timestamp: string;
}

// API Client
export const api = {
  async getQuote(symbol: string): Promise<Quote> {
    const response = await fetch(`${API_BASE_URL}/api/quote/${symbol}`);
    if (!response.ok) throw new Error('Failed to fetch quote');
    return response.json();
  },

  async getVolumeWalls(
    symbol: string,
    expiryDate: string,
    strikeSpacing: number = 5,
    numStrikes: number = 20
  ): Promise<VolumeWallsData> {
    const params = new URLSearchParams({
      expiry_date: expiryDate,
      strike_spacing: strikeSpacing.toString(),
      num_strikes: numStrikes.toString(),
    });
    
    const response = await fetch(
      `${API_BASE_URL}/api/volume-walls/${symbol}?${params}`
    );
    
    if (!response.ok) throw new Error('Failed to fetch volume walls');
    return response.json();
  },

  async getPriceHistory(
    symbol: string,
    frequency: number = 5,
    hours: number = 24
  ): Promise<PriceHistory> {
    const params = new URLSearchParams({
      frequency: frequency.toString(),
      hours: hours.toString(),
    });
    
    const response = await fetch(
      `${API_BASE_URL}/api/price-history/${symbol}?${params}`
    );
    
    if (!response.ok) throw new Error('Failed to fetch price history');
    return response.json();
  },

  async getOptionsChain(
    symbol: string,
    fromDate?: string,
    toDate?: string
  ): Promise<any> {
    const params = new URLSearchParams();
    if (fromDate) params.append('from_date', fromDate);
    if (toDate) params.append('to_date', toDate);
    
    const response = await fetch(
      `${API_BASE_URL}/api/options-chain/${symbol}?${params}`
    );
    
    if (!response.ok) throw new Error('Failed to fetch options chain');
    return response.json();
  },

  async getOptionFinder(
    symbol: string,
    params?: {
      num_expiries?: number;
      option_type?: 'ALL' | 'CALLS' | 'PUTS';
      min_open_interest?: number;
      moneyness_min?: number;
      moneyness_max?: number;
      top_n?: number;
    }
  ): Promise<OptionFinderResponse> {
    const qs = new URLSearchParams();
    if (params?.num_expiries != null) qs.append('num_expiries', String(params.num_expiries));
    if (params?.option_type) qs.append('option_type', params.option_type);
    if (params?.min_open_interest != null) qs.append('min_open_interest', String(params.min_open_interest));
    if (params?.moneyness_min != null) qs.append('moneyness_min', String(params.moneyness_min));
    if (params?.moneyness_max != null) qs.append('moneyness_max', String(params.moneyness_max));
    if (params?.top_n != null) qs.append('top_n', String(params.top_n));

    const response = await fetch(`${API_BASE_URL}/api/option-finder/${symbol}?${qs}`);
    if (!response.ok) throw new Error('Failed to fetch option finder data');
    return response.json();
  },

  async getNetPremiumHeatmap(symbol: string, numExpiries: number = 4): Promise<NetPremiumHeatmapResponse> {
    const params = new URLSearchParams({ num_expiries: String(numExpiries) });
    const response = await fetch(`${API_BASE_URL}/api/net-premium-heatmap/${symbol}?${params}`);
    if (!response.ok) throw new Error('Failed to fetch net premium heatmap');
    return response.json();
  },

  async getBigTrades(symbol: string, minPremium: number = 100000): Promise<any> {
    const params = new URLSearchParams({
      min_premium: minPremium.toString(),
    });
    
    const response = await fetch(
      `${API_BASE_URL}/api/big-trades/${symbol}?${params}`
    );
    
    if (!response.ok) throw new Error('Failed to fetch big trades');
    return response.json();
  },

  async getExpiries(symbol: string): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/api/expiries/${symbol}`);
    if (!response.ok) throw new Error('Failed to fetch expiries');
    const data = await response.json();
    return data.expiries || [];
  },
};

// WebSocket Client
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private messageHandler: ((data: any) => void) | null = null;
  private errorHandler: ((error: Event) => void) | null = null;
  private openHandler: (() => void) | null = null;
  private closeHandler: (() => void) | null = null;

  constructor(endpoint?: string) {
    this.url = endpoint ? `${WS_BASE_URL}${endpoint}` : '';
  }

  onOpen(handler: () => void) {
    this.openHandler = handler;
  }

  onMessage(handler: (data: any) => void) {
    this.messageHandler = handler;
  }

  onError(handler: (error: Event) => void) {
    this.errorHandler = handler;
  }

  onClose(handler: () => void) {
    this.closeHandler = handler;
  }

  connect(endpoint?: string, onMessage?: (data: any) => void, onError?: (error: Event) => void) {
    if (endpoint) {
      this.url = `${WS_BASE_URL}${endpoint}`;
    }
    if (onMessage) {
      this.messageHandler = onMessage;
    }
    if (onError) {
      this.errorHandler = onError;
    }

    if (!this.url) {
      console.error('WebSocket URL is not set');
      return;
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        if (this.openHandler) this.openHandler();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (this.messageHandler) this.messageHandler(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onerror = (error) => {
        if (this.errorHandler) this.errorHandler(error);
      };

      this.ws.onclose = () => {
        if (this.closeHandler) this.closeHandler();
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
        this.connect();
      }, 1000 * this.reconnectAttempts);
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  close() {
    if (this.ws) {
      this.maxReconnectAttempts = 0; // Prevent reconnection
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
