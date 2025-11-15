import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { api, OptionFinderResponse } from '../services/api';
import './OptionFinder.css';

interface GammaData {
  strike: number;
  gamma: number;
  volume: number;
  openInterest: number;
  impliedVolatility: number;
}

const OptionFinder: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [callData, setCallData] = useState<GammaData[]>([]);
  const [putData, setPutData] = useState<GammaData[]>([]);
  const [maxGamma, setMaxGamma] = useState<{ call: GammaData | null; put: GammaData | null }>({
    call: null,
    put: null
  });
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finderData, setFinderData] = useState<OptionFinderResponse | null>(null);

  // Load expiries when symbol changes
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setError(null);
        const list = await api.getExpiries(symbol);
        if (mounted) {
          setExpiries(list);
          setSelectedExpiry(list[0] || '');
        }
      } catch (e: any) {
        if (mounted) setError(e?.message || 'Failed to load expiries');
      }
    })();
    return () => {
      mounted = false;
    };
  }, [symbol]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol.trim() || !selectedExpiry) return;

    setLoading(true);
    setError(null);

    try {
      // Get current price
      const quote = await api.getQuote(symbol);
      const px = (quote.price ?? quote.mark ?? 0) as number;
      setCurrentPrice(px);

      // Get options chain for the selected expiry
      const chainResp = await api.getOptionsChain(symbol, selectedExpiry, selectedExpiry);
      const options = chainResp.options || {};

      // Flatten call/put maps for the chosen expiry
      const callMap = options.callExpDateMap || {};
      const putMap = options.putExpDateMap || {};

      const flatten = (expMap: Record<string, any>): GammaData[] => {
        const out: GammaData[] = [];
        Object.entries(expMap).forEach(([expKey, strikes]) => {
          // expKey like 'YYYY-MM-DD:7' – match prefix with selectedExpiry
          if (!expKey.startsWith(selectedExpiry)) return;
          Object.entries(strikes as Record<string, any[]>).forEach(([strikeStr, contracts]) => {
            if (!contracts || !contracts.length) return;
            const c = contracts[0] as any;
            const strike = parseFloat(strikeStr);
            out.push({
              strike,
              gamma: c.gamma || 0,
              volume: c.totalVolume || 0,
              openInterest: c.openInterest || 0,
              impliedVolatility: (c.volatility || 0) * 100,
            });
          });
        });
        // sort by strike
        return out.sort((a, b) => a.strike - b.strike);
      };

      // Process calls
      const calls: GammaData[] = [];
      let maxCall: GammaData | null = null;
      const flattenedCalls = flatten(callMap);
      flattenedCalls.forEach((d) => {
        calls.push(d);
        if (!maxCall || d.gamma > maxCall.gamma) maxCall = d;
      });

      // Process puts
      const puts: GammaData[] = [];
      let maxPut: GammaData | null = null;
      const flattenedPuts = flatten(putMap);
      flattenedPuts.forEach((d) => {
        puts.push(d);
        if (!maxPut || d.gamma > maxPut.gamma) maxPut = d;
      });

      setCallData(calls);
      setPutData(puts);
      setMaxGamma({ call: maxCall, put: maxPut });

      // Fetch multi-expiry Net GEX heatmap and top lists (defaults)
      try {
        const finder = await api.getOptionFinder(symbol, {
          num_expiries: 5,
          option_type: 'ALL',
          min_open_interest: 0,
          moneyness_min: -50,
          moneyness_max: 50,
          top_n: 5,
        });
        setFinderData(finder);
      } catch (e) {
        // Non-fatal; the base charts still render
        console.warn('Option finder aggregate fetch failed', e);
        setFinderData(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load options data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderGammaChart = () => {
    if (callData.length === 0 && putData.length === 0) return null;

    return (
      <Plot
        data={[
          {
            x: callData.map(d => d.strike),
            y: callData.map(d => d.gamma),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Call Gamma',
            line: { color: '#10b981', width: 2 },
            marker: { size: 6 }
          },
          {
            x: putData.map(d => d.strike),
            y: putData.map(d => d.gamma),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Put Gamma',
            line: { color: '#ef4444', width: 2 },
            marker: { size: 6 }
          },
          {
            x: [currentPrice, currentPrice],
            y: [0, Math.max(...callData.map(d => d.gamma), ...putData.map(d => d.gamma))],
            type: 'scatter',
            mode: 'lines',
            name: 'Current Price',
            line: { color: 'yellow', width: 2, dash: 'dash' }
          }
        ]}
        layout={{
          title: `${symbol} Gamma Exposure`,
          xaxis: { title: 'Strike Price' },
          yaxis: { title: 'Gamma' },
          height: 500,
          plot_bgcolor: '#0f172a',
          paper_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          showlegend: true,
          legend: { x: 0, y: 1 }
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderHeatmap = () => {
    if (!finderData || !finderData.heatmap || !finderData.heatmap.x.length) return null;
    const { x, y, z } = finderData.heatmap;
    return (
      <Plot
        data={[{
          x,
          y: y.map((s) => `$${s}`),
          z,
          type: 'heatmap',
          colorscale: 'RdYlGn',
          zmid: 0,
          hovertemplate: 'Expiry: %{x}<br>Strike: %{y}<br>Net GEX: %{z:.0f}<extra></extra>'
        }]}
        layout={{
          title: `${symbol} Net GEX Heatmap (multi-expiry)`,
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

  const renderTopLists = () => {
    if (!finderData) return null;
    const topCalls = finderData.top_calls || [];
    const topPuts = finderData.top_puts || [];
    return (
      <div className="toplists-grid">
        <div className="toplist">
          <h3>🟢 Top Calls</h3>
          {topCalls.length === 0 && <div className="muted">No data</div>}
          {topCalls.map((r, i) => (
            <div key={`c-${i}`} className="toplist-item">
              <div className="tl-left"><strong>${'{'}r.strike.toFixed(2){'}'}</strong> · {r.expiry}</div>
              <div className="tl-right">{(r.signed_notional_gamma).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
            </div>
          ))}
        </div>
        <div className="toplist">
          <h3>🔴 Top Puts</h3>
          {topPuts.length === 0 && <div className="muted">No data</div>}
          {topPuts.map((r, i) => (
            <div key={`p-${i}`} className="toplist-item">
              <div className="tl-left"><strong>${'{'}r.strike.toFixed(2){'}'}</strong> · {r.expiry}</div>
              <div className="tl-right">{(r.signed_notional_gamma).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="option-finder">
      <div className="page-header">
        <h1>🎯 Max Gamma Finder</h1>
        <p>Discover high gamma exposure strikes and trading opportunities</p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <div className="input-group">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter symbol (e.g., SPY)"
            className="symbol-input"
          />
          <select
            className="expiry-select"
            value={selectedExpiry}
            onChange={(e) => setSelectedExpiry(e.target.value)}
          >
            {expiries.map((exp) => (
              <option key={exp} value={exp}>{exp}</option>
            ))}
          </select>
          <button type="submit" disabled={loading || !symbol} className="search-btn">
            {loading ? 'Searching...' : 'Find Max Gamma'}
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
          <p>Analyzing options chain...</p>
        </div>
      )}

      {!loading && maxGamma.call && maxGamma.put && (
        <>
          <div className="summary-grid">
            <div className="summary-card">
              <div className="summary-label">Current Price</div>
              <div className="summary-value">${currentPrice.toFixed(2)}</div>
            </div>

            <div className="summary-card highlight">
              <div className="summary-label">Max Call Gamma</div>
              <div className="summary-value call-color">
                ${maxGamma.call.strike.toFixed(2)}
              </div>
              <div className="summary-detail">
                Gamma: {maxGamma.call.gamma.toFixed(4)}
              </div>
              <div className="summary-detail">
                Volume: {maxGamma.call.volume.toLocaleString()}
              </div>
              <div className="summary-detail">
                OI: {maxGamma.call.openInterest.toLocaleString()}
              </div>
            </div>

            <div className="summary-card highlight">
              <div className="summary-label">Max Put Gamma</div>
              <div className="summary-value put-color">
                ${maxGamma.put.strike.toFixed(2)}
              </div>
              <div className="summary-detail">
                Gamma: {maxGamma.put.gamma.toFixed(4)}
              </div>
              <div className="summary-detail">
                Volume: {maxGamma.put.volume.toLocaleString()}
              </div>
              <div className="summary-detail">
                OI: {maxGamma.put.openInterest.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="charts-section">
            <div className="chart-container">
              {renderGammaChart()}
            </div>

            <div className="chart-container">
              {renderHeatmap()}
            </div>
          </div>

          {renderTopLists()}

          <div className="info-box">
            <h3>💡 Trading Insights</h3>
            <ul>
              <li>
                <strong>Max Call Gamma Strike:</strong> ${maxGamma.call.strike.toFixed(2)} - 
                This strike has the highest positive gamma, indicating strong dealer hedging activity.
                Price tends to gravitate toward this level.
              </li>
              <li>
                <strong>Max Put Gamma Strike:</strong> ${maxGamma.put.strike.toFixed(2)} - 
                This strike represents maximum negative gamma exposure. 
                Often acts as strong support where dealers buy shares.
              </li>
              <li>
                <strong>Trading Range:</strong> Expect price to oscillate between ${maxGamma.put.strike.toFixed(2)} 
                and ${maxGamma.call.strike.toFixed(2)} due to gamma hedging flows.
              </li>
              <li>
                <strong>Volatility:</strong> High gamma strikes typically see increased trading activity 
                and can act as magnets for price action, especially near expiration.
              </li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default OptionFinder;
