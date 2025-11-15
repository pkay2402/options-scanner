import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { api, VolumeWallsData, NetPremiumHeatmapResponse, PriceHistory } from '../services/api';
import './VolumeWalls.css';

const VolumeWalls: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const [data, setData] = useState<VolumeWallsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expiry, setExpiry] = useState<string>('');
  const [strikeSpacing, setStrikeSpacing] = useState<number>(5);
  const [numStrikes, setNumStrikes] = useState<number>(20);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [showGex, setShowGex] = useState<boolean>(false);
  const [showNetPremium, setShowNetPremium] = useState<boolean>(true);
  const [showVolumeProfile, setShowVolumeProfile] = useState<boolean>(true);
  const [netPremium, setNetPremium] = useState<NetPremiumHeatmapResponse | null>(null);
  const [priceHistory, setPriceHistory] = useState<PriceHistory | null>(null);

  useEffect(() => {
    if (symbol) {
      loadExpiries(symbol);
    }
  }, [symbol]);

  useEffect(() => {
    if (symbol && expiry) {
      loadData();
      // Load supporting datasets in parallel
      loadNetPremium();
      loadPriceHistory();
    }
  }, [symbol, expiry, strikeSpacing, numStrikes]);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await api.getVolumeWalls(symbol, expiry, strikeSpacing, numStrikes);
      setData(result);
    } catch (err: any) {
      setError(err.message || 'Failed to load volume walls data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadExpiries = async (sym: string) => {
    try {
      const exps = await api.getExpiries(sym);
      setExpiries(exps);
      if (!expiry && exps.length > 0) {
        // pick the first (earliest) not in the past
        const today = new Date().toISOString().slice(0, 10);
        const first = exps.find((d) => d >= today) || exps[0];
        setExpiry(first);
      } else if (expiry && !exps.includes(expiry) && exps.length > 0) {
        setExpiry(exps[0]);
      }
    } catch (e) {
      // Fallback: leave expiry empty; user can input manually
      setExpiries([]);
    }
  };

  const loadNetPremium = async () => {
    try {
      const resp = await api.getNetPremiumHeatmap(symbol, 4);
      setNetPremium(resp);
    } catch (e) {
      setNetPremium(null);
    }
  };

  const loadPriceHistory = async () => {
    try {
      const ph = await api.getPriceHistory(symbol, 5, 48);
      setPriceHistory(ph);
    } catch (e) {
      setPriceHistory(null);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadData();
  };

  const renderChart = () => {
    if (!data) return null;
    const strikes = data.all_strikes;
    const callVolumes = strikes.map((s) => data.call_volumes[String(s)] || 0);
    const putVolumes = strikes.map((s) => data.put_volumes[String(s)] || 0);
    const gex = strikes.map((s) => data.gex_by_strike[String(s)] || 0);

    const traces: any[] = [
      <Plot
        data={[]}
        layout={{
          title: `${symbol} Volume Walls (${data.expiry_date})`,
          xaxis: { title: 'Strike Price' },
          yaxis: { title: 'Volume', zeroline: true },
          yaxis2: showGex ? { title: 'GEX', overlaying: 'y', side: 'right' } : undefined,
          barmode: 'overlay',
          height: 500,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: true,
          legend: { x: 0, y: 1 },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    ];

    const dataTraces: any[] = [
      {
        x: strikes,
        y: callVolumes,
        type: 'bar',
        name: 'Call Volume',
        marker: { color: 'rgba(16, 185, 129, 0.7)' },
      },
      {
        x: strikes,
        y: putVolumes.map((v: number) => -v),
        type: 'bar',
        name: 'Put Volume',
        marker: { color: 'rgba(239, 68, 68, 0.7)' },
      },
      {
        x: [data.underlying_price, data.underlying_price],
        y: [Math.min(...putVolumes.map((v: number) => -v)), Math.max(...callVolumes)],
        type: 'scatter',
        mode: 'lines',
        name: 'Current Price',
        line: { color: 'yellow', width: 2, dash: 'dash' },
      },
    ];

    if (showGex) {
      dataTraces.push({
        x: strikes,
        y: gex,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'GEX',
        line: { color: '#60a5fa', width: 2 },
        yaxis: 'y2',
      });
    }

    return (
      <Plot
        data={dataTraces}
        layout={{
          title: `${symbol} Volume Walls (${data.expiry_date})`,
          xaxis: { title: 'Strike Price' },
          yaxis: { title: 'Volume', zeroline: true },
          yaxis2: showGex ? { title: 'GEX', overlaying: 'y', side: 'right' } : undefined,
          barmode: 'overlay',
          height: 500,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: true,
          legend: { x: 0, y: 1 },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderNetPremiumHeatmap = () => {
    if (!netPremium || !netPremium.heatmap.x.length) return null;
    const { x, y, z } = netPremium.heatmap;
    return (
      <Plot
        data={[{
          x,
          y: y.map((s) => `$${s}`),
          z,
          type: 'heatmap',
          colorscale: [
            [0.0, '#d32f2f'],
            [0.25, '#ef5350'],
            [0.4, '#ffcdd2'],
            [0.5, '#ffffff'],
            [0.6, '#c8e6c9'],
            [0.75, '#66bb6a'],
            [1.0, '#2e7d32']
          ],
          zmid: 0,
          hovertemplate: 'Expiry: %{x}<br>Strike: %{y}<br>Net Premium: %{z:.0f}<extra></extra>'
        }]}
        layout={{
          title: `${symbol} Net Premium Heatmap (Call - Put)`,
          xaxis: { title: 'Expiration Date' },
          yaxis: { title: 'Strike' },
          height: 500,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: false,
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderVolumeProfile = () => {
    if (!data) return null;
    const up = data.underlying_price;
    const all = data.all_strikes;
    if (!all || all.length === 0) return null;
    // Dynamic range similar to Streamlit version
    let rangeBuffer = 10;
    if (up < 100) rangeBuffer = up * 0.15;
    else if (up < 500) rangeBuffer = up * 0.1;
    else rangeBuffer = up * 0.08;

    const minS = up - rangeBuffer;
    const maxS = up + rangeBuffer;
    const strikes = all.filter((s) => s >= minS && s <= maxS);
    const sortedStrikes = strikes.length ? strikes.sort((a, b) => a - b) : all.slice().sort((a, b) => a - b).slice(0, 20);

    const net = sortedStrikes.map((s) => data.net_volumes[String(s)] || 0);
    const colors = net.map((v) => (v > 0 ? '#ef4444' : '#22c55e'));

    return (
      <Plot
        data={[{
          y: sortedStrikes,
          x: net,
          type: 'bar',
          orientation: 'h',
          name: 'Net Volume (Put - Call)',
          marker: { color: colors, line: { color: 'rgba(0,0,0,0.3)', width: 0.5 } },
          hovertemplate: 'Strike: $%{y}<br>Net Volume: %{x:,.0f}<extra></extra>'
        }]}
        layout={{
          title: `${symbol} Net Option Volume Profile by Strike`,
          xaxis: { title: 'Net Volume (Put - Call)' },
          yaxis: { title: 'Strike Price ($)' },
          height: 700,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          shapes: [
            data.net_call_wall?.strike ? {
              type: 'line', x0: Math.min(...net), x1: Math.max(...net), y0: data.net_call_wall!.strike, y1: data.net_call_wall!.strike,
              line: { color: 'darkgreen', dash: 'dot' }
            } : undefined,
            data.net_put_wall?.strike ? {
              type: 'line', x0: Math.min(...net), x1: Math.max(...net), y0: data.net_put_wall!.strike, y1: data.net_put_wall!.strike,
              line: { color: 'darkred', dash: 'dot' }
            } : undefined,
            data.flip_level ? {
              type: 'line', x0: Math.min(...net), x1: Math.max(...net), y0: data.flip_level!, y1: data.flip_level!,
              line: { color: 'purple', dash: 'dash' }
            } : undefined,
          ].filter(Boolean) as any,
          showlegend: false,
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderIntradayChart = () => {
    if (!priceHistory || !priceHistory.candles?.length || !data) return null;
    const candles = priceHistory.candles;
    // Convert to arrays
    const x = candles.map(c => new Date(c.datetime));
    const open = candles.map(c => c.open);
    const high = candles.map(c => c.high);
    const low = candles.map(c => c.low);
    const close = candles.map(c => c.close);
    const vol = candles.map(c => c.volume);

    // Compute EMA21
    const ema = (vals: number[], span: number) => {
      const k = 2 / (span + 1);
      let prev = vals[0];
      const out = [prev];
      for (let i = 1; i < vals.length; i++) {
        prev = vals[i] * k + prev * (1 - k);
        out.push(prev);
      }
      return out;
    };
    const ema21 = ema(close, 21);

    // VWAP (simple rolling cumulative using typical price)
    const typ = candles.map(c => (c.high + c.low + c.close) / 3);
    let cumPV = 0, cumV = 0;
    const vwap = typ.map((p, i) => { cumPV += p * vol[i]; cumV += vol[i]; return cumPV / Math.max(1, cumV); });

    const shapes: any[] = [];
    if (data.call_wall?.strike) shapes.push({ type: 'line', x0: x[0], x1: x[x.length-1], y0: data.call_wall.strike, y1: data.call_wall.strike, line: { color: '#22c55e', dash: 'dot', width: 2 } });
    if (data.put_wall?.strike) shapes.push({ type: 'line', x0: x[0], x1: x[x.length-1], y0: data.put_wall.strike, y1: data.put_wall.strike, line: { color: '#ef4444', dash: 'dot', width: 2 } });
    if (data.flip_level) shapes.push({ type: 'line', x0: x[0], x1: x[x.length-1], y0: data.flip_level, y1: data.flip_level, line: { color: '#a855f7', dash: 'solid', width: 2 } });

    return (
      <Plot
        data={[
          { x, open, high, low, close, type: 'candlestick', name: 'Price', increasing: { line: { color: '#26a69a' } }, decreasing: { line: { color: '#ef5350' } } },
          { x, y: vwap, type: 'scatter', mode: 'lines', name: 'VWAP', line: { color: '#00bcd4', width: 2.5 } },
          { x, y: ema21, type: 'scatter', mode: 'lines', name: 'EMA 21', line: { color: '#ff9800', width: 2 } },
        ]}
        layout={{
          title: `${symbol} Intraday + Walls`,
          xaxis: { title: 'Time (ET)' },
          yaxis: { title: 'Price ($)' },
          height: 550,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: true,
          shapes,
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderBiasBanner = () => {
    if (!data) return null;
    const net = data.totals.net_vol;
    let color = '#2196f3';
    let text = '🐂 MILD BULLISH BIAS';
    if (net > 10000) { color = '#f44336'; text = '🐻 STRONG BEARISH BIAS'; }
    else if (net > 0) { color = '#ff9800'; text = '🐻 MILD BEARISH BIAS'; }
    else if (net < -10000) { color = '#4caf50'; text = '🐂 STRONG BULLISH BIAS'; }
    return (
      <div style={{
        background: `linear-gradient(90deg, ${color} 0%, ${color}cc 100%)`,
        color: '#fff', padding: '12px 16px', borderRadius: 8, textAlign: 'center',
        fontSize: 18, fontWeight: 800, marginBottom: 12
      }}>{text}</div>
    );
  };

  const renderAlerts = () => {
    if (!data) return null;
    const alerts: { priority: string; type: string; message: string; action?: string }[] = [];
    const up = data.underlying_price;
    const putW = data.put_wall?.strike ?? null;
    const callW = data.call_wall?.strike ?? null;
    const flip = data.flip_level ?? null;

    if (putW) {
      const dist = ((up - putW) / putW) * 100;
      if (0 <= dist && dist <= 1) alerts.push({ priority: '🔴 HIGH', type: 'Support Test', message: `Testing Put Wall ${putW.toFixed(2)} (${dist.toFixed(2)}% above)` });
      else if (-0.5 <= dist && dist < 0) alerts.push({ priority: '🔴 HIGH', type: 'Support Break', message: `Broke below Put Wall ${putW.toFixed(2)}` });
    }
    if (callW) {
      const dist = ((callW - up) / up) * 100;
      if (0 <= dist && dist <= 1) alerts.push({ priority: '🔴 HIGH', type: 'Resistance Test', message: `Approaching Call Wall ${callW.toFixed(2)} (${dist.toFixed(2)}% below)` });
      else if (-0.5 <= dist && dist < 0) alerts.push({ priority: '🔴 HIGH', type: 'Resistance Break', message: `Broke above Call Wall ${callW.toFixed(2)}` });
    }
    if (flip) {
      const dist = Math.abs((up - flip) / flip) * 100;
      if (dist <= 0.5) alerts.push({ priority: '🟡 MEDIUM', type: 'Flip Level', message: `Near Flip ${flip.toFixed(2)} (${dist.toFixed(2)}%)` });
    }
    if (!alerts.length) return null;
    return (
      <div className="alerts-list">
        {alerts.map((a, i) => (
          <div key={i} className="alert-item"><strong>{a.priority}</strong> {a.type} — {a.message}</div>
        ))}
      </div>
    );
  };

  const formatNumber = (num: number) => num.toLocaleString();
  const formatPrice = (price: number) => (price ?? 0).toFixed(2);

  const expiryOptions = expiries;

  return (
    <div className="volume-walls">
      <div className="page-header">
        <h1>📊 Volume Walls</h1>
        <p>Identify key support and resistance levels from options volume</p>
      </div>

      <form onSubmit={handleSubmit} className="search-form">
        <div className="input-group">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter symbol (e.g., SPY)"
            className="symbol-input"
          />
          {expiryOptions.length > 0 ? (
            <select className="expiry-input" value={expiry} onChange={(e) => setExpiry(e.target.value)}>
              {expiryOptions.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              className="expiry-input"
              placeholder="YYYY-MM-DD"
              value={expiry}
              onChange={(e) => setExpiry(e.target.value)}
            />
          )}
          <input
            type="number"
            className="spacing-input"
            value={strikeSpacing}
            min={1}
            step={1}
            onChange={(e) => setStrikeSpacing(parseInt(e.target.value || '5'))}
            title="Strike spacing"
          />
          <input
            type="number"
            className="count-input"
            value={numStrikes}
            min={5}
            step={1}
            onChange={(e) => setNumStrikes(parseInt(e.target.value || '20'))}
            title="# strikes each side"
          />
          <button type="submit" disabled={loading || !symbol} className="search-btn">
            {loading ? 'Loading...' : 'Search'}
          </button>
        </div>
      </form>

      {error && (
        <div className="alert alert-danger">
          {error}
        </div>
      )}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading volume walls data...</p>
        </div>
      )}

      {data && !loading && (
        <>
          {renderBiasBanner()}
          <div className="summary-grid">
            <div className="summary-card">
              <div className="summary-label">Current Price</div>
              <div className="summary-value">${formatPrice(data.underlying_price)}</div>
            </div>

            <div className="summary-card highlight">
              <div className="summary-label">Call Wall</div>
              <div className="summary-value call-color">
                ${formatPrice(data.call_wall?.strike ?? 0)}
              </div>
              <div className="summary-detail">
                Volume: {formatNumber(data.call_wall?.volume ?? 0)}
              </div>
            </div>

            <div className="summary-card highlight">
              <div className="summary-label">Put Wall</div>
              <div className="summary-value put-color">
                ${formatPrice(data.put_wall?.strike ?? 0)}
              </div>
              <div className="summary-detail">
                Volume: {formatNumber(data.put_wall?.volume ?? 0)}
              </div>
            </div>

            <div className="summary-card">
              <div className="summary-label">Flip Level</div>
              <div className="summary-value flip-color">
                ${formatPrice(data.flip_level ?? 0)}
              </div>
              <div className="summary-detail">
                {data.flip_level && data.underlying_price > data.flip_level ? '🟢 Bullish' : '🔴 Bearish'}
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, marginBottom: 8 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input type="checkbox" checked={showGex} onChange={(e) => setShowGex(e.target.checked)} />
                Show GEX overlay
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input type="checkbox" checked={showNetPremium} onChange={(e) => setShowNetPremium(e.target.checked)} />
                Show Net Premium Heatmap
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input type="checkbox" checked={showVolumeProfile} onChange={(e) => setShowVolumeProfile(e.target.checked)} />
                Show Volume Profile
              </label>
            </div>
            {renderChart()}
          </div>

          <div className="chart-container">
            {renderIntradayChart()}
          </div>

          {showNetPremium && (
            <div className="chart-container">
              {renderNetPremiumHeatmap()}
            </div>
          )}

          {showVolumeProfile && (
            <div className="chart-container">
              {renderVolumeProfile()}
            </div>
          )}

          {renderAlerts()}

          <div className="info-box">
            <h3>💡 Understanding Volume Walls</h3>
            <ul>
              <li>
                <strong>Call Wall ({formatPrice(data.call_wall?.strike ?? 0)}):</strong> Highest call volume strike above current price. 
                Acts as resistance where dealers may hedge by selling shares.
              </li>
              <li>
                <strong>Put Wall ({formatPrice(data.put_wall?.strike ?? 0)}):</strong> Highest put volume strike below current price. 
                Acts as support where dealers may hedge by buying shares.
              </li>
              <li>
                <strong>Flip Level ({formatPrice(data.flip_level ?? 0)}):</strong> Strike where net dealer positioning changes. 
                {data.flip_level && data.underlying_price > data.flip_level 
                  ? ' Currently ABOVE - bullish bias (dealers buy on dips).' 
                  : ' Currently BELOW - bearish bias (dealers sell on rips).'}
              </li>
              <li>
                <strong>Trading Range:</strong> Stock tends to gravitate between put wall (${formatPrice(data.put_wall?.strike ?? 0)}) 
                and call wall (${formatPrice(data.call_wall?.strike ?? 0)}).
              </li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default VolumeWalls;

