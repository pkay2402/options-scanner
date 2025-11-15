import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import './Dashboard.css';

interface MarketOverview {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
}

const Dashboard: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketOverview[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const symbols = ['SPY', 'QQQ', 'IWM', 'DIA', '$VIX'];

  useEffect(() => {
    loadMarketData();
    const interval = setInterval(loadMarketData, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, []);

  const loadMarketData = async () => {
    try {
      setLoading(true);
      const data: MarketOverview[] = [];
      
      for (const symbol of symbols) {
        try {
          const quote = await api.getQuote(symbol);
          data.push({
            symbol,
            price: quote.mark ?? quote.price ?? 0,
            change: quote.change ?? 0,
            changePercent: quote.changePercent ?? 0
          });
        } catch (err) {
          console.error(`Error loading ${symbol}:`, err);
          // Add placeholder data for failed symbols
          data.push({
            symbol,
            price: 0,
            change: 0,
            changePercent: 0
          });
        }
      }
      
      setMarketData(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError('Failed to load market data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-success';
    if (change < 0) return 'text-danger';
    return 'text-muted';
  };

  const formatPrice = (price: number | undefined | null) => {
    if (price === undefined || price === null || isNaN(price)) return '0.00';
    return price.toFixed(2);
  };
  
  const formatChange = (change: number | undefined | null) => {
    if (change === undefined || change === null || isNaN(change)) return '+0.00';
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}`;
  };
  
  const formatPercent = (percent: number | undefined | null) => {
    if (percent === undefined || percent === null || isNaN(percent)) return '+0.00%';
    const sign = percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  };

  if (loading && marketData.length === 0) {
    return (
      <div className="dashboard">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading market data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Market Overview</h1>
        <button onClick={loadMarketData} className="refresh-btn" disabled={loading}>
          {loading ? '⟳' : '↻'} Refresh
        </button>
      </div>

      {error && (
        <div className="alert alert-danger">
          {error}
        </div>
      )}

      <div className="last-update">
        Last updated: {lastUpdate.toLocaleTimeString()}
      </div>

      <div className="market-grid">
        {marketData.map((item) => (
          <div key={item.symbol} className="market-card">
            <div className="market-symbol">{item.symbol}</div>
            <div className="market-price">${formatPrice(item.price)}</div>
            <div className={`market-change ${getChangeColor(item.change)}`}>
              <span className="change-value">{formatChange(item.change)}</span>
              <span className="change-percent">{formatPercent(item.changePercent)}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="dashboard-info">
        <div className="info-card">
          <h3>🎯 Quick Actions</h3>
          <ul>
            <li>
              <a href="/volume-walls">View Volume Walls</a> - Identify key support/resistance levels
            </li>
            <li>
              <a href="/flow-scanner">Monitor Options Flow</a> - Track big money moves in real-time
            </li>
            <li>
              <a href="/option-finder">Find Max Gamma</a> - Discover high gamma opportunities
            </li>
            <li>
              <a href="/alerts">Configure Alerts</a> - Set up custom notifications
            </li>
          </ul>
        </div>

        <div className="info-card">
          <h3>📊 Features</h3>
          <ul>
            <li>Real-time options flow monitoring</li>
            <li>Volume walls and flip level detection</li>
            <li>Max gamma scanner for high-impact strikes</li>
            <li>Push notifications for big trades</li>
            <li>Works offline with service worker</li>
          </ul>
        </div>

        <div className="info-card">
          <h3>💡 Tips</h3>
          <ul>
            <li>Enable notifications for instant alerts</li>
            <li>Add to home screen for app-like experience</li>
            <li>Use landscape mode for better chart viewing</li>
            <li>Refresh data by pulling down on mobile</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
