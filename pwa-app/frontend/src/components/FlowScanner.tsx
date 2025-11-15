import React, { useState, useEffect, useRef } from 'react';
import { WebSocketClient } from '../services/api';
import { NotificationService } from '../services/notifications';
import './FlowScanner.css';

interface FlowTrade {
  timestamp: string;
  symbol: string;
  strike: number;
  expiration: string;
  type: 'CALL' | 'PUT';
  side: 'BUY' | 'SELL';
  volume: number;
  premium: number;
  spot_price: number;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
}

const FlowScanner: React.FC = () => {
  const [trades, setTrades] = useState<FlowTrade[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [minPremium, setMinPremium] = useState(100000); // $100k minimum
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['SPY', 'QQQ', 'AAPL']);
  const [newSymbol, setNewSymbol] = useState('');
  const wsRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocketClient('/ws/flow');
    
    ws.onOpen(() => {
      setIsConnected(true);
      console.log('Connected to options flow');
    });

    ws.onMessage((data) => {
      const trade = data as FlowTrade;
      
      // Filter by premium threshold
      if (trade.premium >= minPremium) {
        setTrades(prev => [trade, ...prev].slice(0, 100)); // Keep last 100 trades
        
        // Send notification for big trades
        if (trade.premium >= 500000) { // $500k+
          NotificationService.sendAlert(
            '🚨 Big Trade Alert',
            `${trade.symbol} ${trade.strike}${trade.type[0]} - $${formatMoney(trade.premium)}`,
            [
              { action: 'view', title: 'View Details' },
              { action: 'close', title: 'Dismiss' }
            ]
          );
        }
      }
    });

    ws.onClose(() => {
      setIsConnected(false);
      console.log('Disconnected from options flow');
      // Reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    });

    ws.onError((error) => {
      console.error('WebSocket error:', error);
    });

    ws.connect();
    wsRef.current = ws;
  };

  const addSymbol = () => {
    const symbol = newSymbol.toUpperCase().trim();
    if (symbol && !selectedSymbols.includes(symbol)) {
      setSelectedSymbols([...selectedSymbols, symbol]);
      setNewSymbol('');
    }
  };

  const removeSymbol = (symbol: string) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
  };

  const formatMoney = (amount: number) => {
    if (amount >= 1000000) {
      return `${(amount / 1000000).toFixed(2)}M`;
    }
    return `${(amount / 1000).toFixed(0)}K`;
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getSentimentBadge = (sentiment: string) => {
    const classes = {
      BULLISH: 'badge-success',
      BEARISH: 'badge-danger',
      NEUTRAL: 'badge-secondary'
    };
    return classes[sentiment as keyof typeof classes] || 'badge-secondary';
  };

  const getTypeBadge = (type: string) => {
    return type === 'CALL' ? 'badge-call' : 'badge-put';
  };

  const filteredTrades = trades.filter(t => 
    selectedSymbols.length === 0 || selectedSymbols.includes(t.symbol)
  );

  return (
    <div className="flow-scanner">
      <div className="page-header">
        <h1>🌊 Options Flow Scanner</h1>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Live' : '🔴 Disconnected'}
          </span>
        </div>
      </div>

      <div className="filters-section">
        <div className="filter-group">
          <label>Minimum Premium</label>
          <select 
            value={minPremium} 
            onChange={(e) => setMinPremium(Number(e.target.value))}
            className="filter-select"
          >
            <option value="50000">$50K+</option>
            <option value="100000">$100K+</option>
            <option value="250000">$250K+</option>
            <option value="500000">$500K+</option>
            <option value="1000000">$1M+</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Filter by Symbols ({selectedSymbols.length})</label>
          <div className="symbol-chips">
            {selectedSymbols.map(symbol => (
              <div key={symbol} className="chip">
                {symbol}
                <button onClick={() => removeSymbol(symbol)} className="chip-remove">×</button>
              </div>
            ))}
            <div className="add-symbol-group">
              <input
                type="text"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                placeholder="Add symbol"
                className="symbol-input-small"
              />
              <button onClick={addSymbol} className="add-btn">+</button>
            </div>
          </div>
        </div>
      </div>

      <div className="stats-bar">
        <div className="stat">
          <span className="stat-label">Total Trades</span>
          <span className="stat-value">{filteredTrades.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Total Premium</span>
          <span className="stat-value">
            ${formatMoney(filteredTrades.reduce((sum, t) => sum + t.premium, 0))}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Bullish</span>
          <span className="stat-value text-success">
            {filteredTrades.filter(t => t.sentiment === 'BULLISH').length}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Bearish</span>
          <span className="stat-value text-danger">
            {filteredTrades.filter(t => t.sentiment === 'BEARISH').length}
          </span>
        </div>
      </div>

      <div className="trades-container">
        {filteredTrades.length === 0 ? (
          <div className="empty-state">
            <p>Waiting for trades...</p>
            <p className="text-secondary">
              Big money options trades will appear here in real-time
            </p>
          </div>
        ) : (
          <div className="trades-list">
            {filteredTrades.map((trade, idx) => (
              <div key={idx} className="trade-card">
                <div className="trade-header">
                  <div className="trade-symbol">{trade.symbol}</div>
                  <div className="trade-time">{formatTime(trade.timestamp)}</div>
                </div>
                <div className="trade-details">
                  <div className="trade-option">
                    ${trade.strike} {trade.type}
                    <span className={`badge ${getTypeBadge(trade.type)}`}>
                      {trade.type}
                    </span>
                  </div>
                  <div className="trade-expiry">{trade.expiration}</div>
                </div>
                <div className="trade-metrics">
                  <div className="metric">
                    <span className="metric-label">Premium</span>
                    <span className="metric-value premium">
                      ${formatMoney(trade.premium)}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Volume</span>
                    <span className="metric-value">{trade.volume.toLocaleString()}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Side</span>
                    <span className={`metric-value ${trade.side === 'BUY' ? 'text-success' : 'text-danger'}`}>
                      {trade.side}
                    </span>
                  </div>
                </div>
                <div className="trade-footer">
                  <span className="spot-price">Spot: ${trade.spot_price.toFixed(2)}</span>
                  <span className={`badge ${getSentimentBadge(trade.sentiment)}`}>
                    {trade.sentiment}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default FlowScanner;
