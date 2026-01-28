'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { cn, formatPrice, formatCurrency } from '@/lib/utils';
import { Zap, TrendingUp, TrendingDown, RefreshCw, Clock, AlertTriangle } from 'lucide-react';

interface ZeroDTEData {
  symbol: string;
  underlyingPrice: number;
  callWall: number;
  putWall: number;
  maxPain: number;
  totalGex: number;
  flipPrice: number;
  topStrikes: {
    strike: number;
    callVol: number;
    putVol: number;
    callOI: number;
    putOI: number;
    netGamma: number;
  }[];
  ivRank: number;
  expectedMove: number;
}

const ZERO_DTE_SYMBOLS = ['SPY', 'QQQ', 'IWM'];

async function fetch0DTEData(symbol: string): Promise<ZeroDTEData | null> {
  try {
    // Get today's date for 0DTE
    const today = new Date().toISOString().split('T')[0];
    
    const [wallsRes, gexRes] = await Promise.all([
      fetch(`/api/volume-walls?symbol=${symbol}&expiry=${today}`),
      fetch(`/api/gex?symbol=${symbol}`),
    ]);
    
    const walls = wallsRes.ok ? await wallsRes.json() : null;
    const gex = gexRes.ok ? await gexRes.json() : null;
    
    if (!walls && !gex) return null;
    
    // Calculate max pain (strike with least value to option holders)
    const topStrikes = (walls?.walls || [])
      .sort((a: {callOI: number, putOI: number}, b: {callOI: number, putOI: number}) => 
        (b.callOI + b.putOI) - (a.callOI + a.putOI))
      .slice(0, 10);
    
    // Estimate max pain from put/call wall midpoint
    const maxPain = walls ? (walls.callWall + walls.putWall) / 2 : gex?.flipPrice || 0;
    
    return {
      symbol,
      underlyingPrice: walls?.underlyingPrice || gex?.underlyingPrice || 0,
      callWall: walls?.callWall || 0,
      putWall: walls?.putWall || 0,
      maxPain,
      totalGex: gex?.totalGex || 0,
      flipPrice: gex?.flipPrice || 0,
      topStrikes,
      ivRank: 45 + Math.random() * 30, // Placeholder - would need IV data
      expectedMove: gex?.expectedMove || 0,
    };
  } catch (error) {
    console.error(`0DTE fetch error for ${symbol}:`, error);
    return null;
  }
}

function useZeroDTE(symbol: string) {
  return useQuery({
    queryKey: ['0dte', symbol],
    queryFn: () => fetch0DTEData(symbol),
    staleTime: 2 * 60 * 1000,
    refetchInterval: 3 * 60 * 1000, // Rate limit friendly
    enabled: !!symbol,
  });
}

export function ZeroDTEPanel() {
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const { data, isLoading, refetch, isFetching } = useZeroDTE(selectedSymbol);
  
  // Time until market close
  const [timeToClose, setTimeToClose] = useState('');
  
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const close = new Date();
      close.setHours(16, 0, 0, 0); // 4 PM
      
      if (now > close) {
        setTimeToClose('Market Closed');
        return;
      }
      
      const diff = close.getTime() - now.getTime();
      const hours = Math.floor(diff / (1000 * 60 * 60));
      const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      setTimeToClose(`${hours}h ${mins}m to close`);
    };
    
    updateTime();
    const interval = setInterval(updateTime, 60000);
    return () => clearInterval(interval);
  }, []);

  const priceVsMaxPain = data 
    ? ((data.underlyingPrice - data.maxPain) / data.maxPain * 100).toFixed(2)
    : 0;
  
  const gexBias = data?.totalGex && data.totalGex > 0 ? 'Positive (Pinning)' : 'Negative (Volatile)';

  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          <h3 className="text-white font-semibold">⚡ 0DTE Dashboard</h3>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Symbol selector */}
          <div className="flex gap-1 bg-slate-700 rounded p-0.5">
            {ZERO_DTE_SYMBOLS.map((sym) => (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className={cn(
                  "px-2 py-1 text-xs rounded transition-colors",
                  selectedSymbol === sym 
                    ? "bg-yellow-500 text-black font-semibold" 
                    : "text-slate-400 hover:text-white"
                )}
              >
                {sym}
              </button>
            ))}
          </div>
          
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="p-1.5 hover:bg-slate-700 rounded"
          >
            <RefreshCw className={cn("w-4 h-4 text-slate-400", isFetching && "animate-spin")} />
          </button>
        </div>
      </div>

      {/* Time warning */}
      <div className="px-4 py-2 bg-yellow-500/10 border-b border-yellow-500/20 flex items-center gap-2">
        <Clock className="w-4 h-4 text-yellow-400" />
        <span className="text-yellow-400 text-sm font-medium">{timeToClose}</span>
        <span className="text-slate-400 text-xs ml-auto">0DTE = Today&apos;s Expiry Only</span>
      </div>

      {/* Content */}
      <div className="p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-48">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : !data ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-500">
            <AlertTriangle className="w-8 h-8 mb-2" />
            <span>No 0DTE data available</span>
            <span className="text-xs mt-1">May be outside market hours</span>
          </div>
        ) : (
          <>
            {/* Price & Key Levels */}
            <div className="grid grid-cols-4 gap-3 mb-4">
              <div className="bg-slate-700/50 rounded p-3">
                <div className="text-slate-400 text-xs mb-1">Current Price</div>
                <div className="text-white font-mono text-lg">${formatPrice(data.underlyingPrice)}</div>
              </div>
              <div className="bg-slate-700/50 rounded p-3">
                <div className="text-slate-400 text-xs mb-1">Max Pain</div>
                <div className="text-amber-400 font-mono text-lg">${formatPrice(data.maxPain)}</div>
                <div className={cn(
                  "text-xs mt-1",
                  Number(priceVsMaxPain) > 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {Number(priceVsMaxPain) > 0 ? '+' : ''}{priceVsMaxPain}% from max pain
                </div>
              </div>
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-3">
                <div className="text-emerald-400 text-xs mb-1 flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" /> Call Wall
                </div>
                <div className="text-emerald-400 font-mono text-lg">${formatPrice(data.callWall)}</div>
              </div>
              <div className="bg-red-500/10 border border-red-500/30 rounded p-3">
                <div className="text-red-400 text-xs mb-1 flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" /> Put Wall
                </div>
                <div className="text-red-400 font-mono text-lg">${formatPrice(data.putWall)}</div>
              </div>
            </div>

            {/* GEX & Gamma Info */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-700/50 rounded p-3">
                <div className="text-slate-400 text-xs mb-1">Total GEX</div>
                <div className={cn(
                  "font-mono text-lg",
                  data.totalGex >= 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {data.totalGex >= 0 ? '+' : ''}{(data.totalGex / 1e9).toFixed(2)}B
                </div>
              </div>
              <div className="bg-slate-700/50 rounded p-3">
                <div className="text-slate-400 text-xs mb-1">Gamma Flip</div>
                <div className="text-purple-400 font-mono text-lg">${formatPrice(data.flipPrice)}</div>
              </div>
              <div className="bg-slate-700/50 rounded p-3">
                <div className="text-slate-400 text-xs mb-1">GEX Bias</div>
                <div className={cn(
                  "font-semibold text-sm",
                  data.totalGex >= 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {gexBias}
                </div>
              </div>
            </div>

            {/* Top Strikes Table */}
            <div className="bg-slate-700/30 rounded overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-600 text-slate-400 text-xs font-medium">
                Top Volume Strikes (0DTE)
              </div>
              <div className="max-h-[180px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-700/50 sticky top-0">
                    <tr className="text-xs text-slate-400">
                      <th className="px-3 py-2 text-left">Strike</th>
                      <th className="px-3 py-2 text-right">Call Vol</th>
                      <th className="px-3 py-2 text-right">Put Vol</th>
                      <th className="px-3 py-2 text-right">Net γ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.topStrikes.map((strike, i) => (
                      <tr key={i} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                        <td className="px-3 py-2 font-mono text-white">
                          ${strike.strike}
                          {strike.strike === data.callWall && (
                            <span className="ml-1 text-emerald-400 text-xs">C</span>
                          )}
                          {strike.strike === data.putWall && (
                            <span className="ml-1 text-red-400 text-xs">P</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-emerald-400">
                          {strike.callVol?.toLocaleString() || '-'}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-red-400">
                          {strike.putVol?.toLocaleString() || '-'}
                        </td>
                        <td className={cn(
                          "px-3 py-2 text-right font-mono",
                          strike.netGamma >= 0 ? "text-emerald-400" : "text-red-400"
                        )}>
                          {strike.netGamma >= 0 ? '+' : ''}{(strike.netGamma / 1e6).toFixed(1)}M
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Interpretation */}
            <div className="mt-4 p-3 bg-slate-700/30 rounded text-xs text-slate-400">
              <span className="text-slate-300 font-medium">0DTE Interpretation: </span>
              {data.totalGex >= 0 ? (
                <span>Positive GEX suggests price pinning between ${formatPrice(data.putWall)}-${formatPrice(data.callWall)}. Expect mean reversion.</span>
              ) : (
                <span>Negative GEX below ${formatPrice(data.flipPrice)} = potential for increased volatility. Breakouts more likely.</span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
