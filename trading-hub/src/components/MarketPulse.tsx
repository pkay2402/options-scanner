'use client';

import { useState, useEffect } from 'react';
import { useMarketPulse, useWhaleFlows } from '@/hooks/useMarketData';
import { formatPrice, formatPercent, formatCurrency, cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Activity, Anchor } from 'lucide-react';

function ClientTime() {
  const [time, setTime] = useState<string>('');
  
  useEffect(() => {
    const update = () => {
      setTime(new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true 
      }));
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);
  
  return <div className="text-slate-500 text-xs font-mono">{time}</div>;
}

export function MarketPulse() {
  const { data: pulse, isLoading: pulseLoading } = useMarketPulse();
  const { data: whaleFlows } = useWhaleFlows();

  const topWhales = whaleFlows?.slice(0, 5) || [];

  return (
    <div className="bg-slate-900 border-b border-slate-700 px-4 py-2">
      <div className="flex items-center justify-between gap-6">
        {/* Market Indices */}
        <div className="flex items-center gap-6">
          {/* SPY */}
          <MarketIndex
            symbol="SPY"
            price={pulse?.spy?.lastPrice}
            change={pulse?.spy?.netPercentChange}
            loading={pulseLoading}
          />
          
          {/* QQQ */}
          <MarketIndex
            symbol="QQQ"
            price={pulse?.qqq?.lastPrice}
            change={pulse?.qqq?.netPercentChange}
            loading={pulseLoading}
          />
          
          {/* VIX */}
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-amber-400" />
            <span className="text-slate-400 text-sm">VIX</span>
            {pulseLoading ? (
              <span className="text-slate-500 text-sm">--</span>
            ) : (
              <span className={cn(
                "font-mono text-sm font-medium",
                (pulse?.vix?.lastPrice || 0) > 20 ? "text-red-400" : "text-emerald-400"
              )}>
                {formatPrice(pulse?.vix?.lastPrice || 0, 1)}
              </span>
            )}
          </div>
        </div>

        {/* Whale Ticker */}
        <div className="flex-1 overflow-hidden">
          <div className="flex items-center gap-4 animate-marquee">
            <Anchor className="w-4 h-4 text-blue-400 flex-shrink-0" />
            <span className="text-slate-400 text-xs font-medium">WHALE FLOW:</span>
            {topWhales.length > 0 ? (
              topWhales.map((whale, idx) => (
                <div key={idx} className="flex items-center gap-1 text-xs whitespace-nowrap">
                  <span className="text-white font-medium">{whale.symbol}</span>
                  <span className={cn(
                    "font-mono",
                    whale.type === 'CALL' ? "text-emerald-400" : "text-red-400"
                  )}>
                    {whale.strike}{whale.type === 'CALL' ? 'C' : 'P'}
                  </span>
                  <span className="text-slate-500">
                    {whale.volOiRatio.toFixed(1)}x
                  </span>
                  <span className="text-amber-400">
                    {formatCurrency(whale.premium)}
                  </span>
                  <span className="text-slate-600 mx-2">|</span>
                </div>
              ))
            ) : (
              <span className="text-slate-500 text-xs">Loading flows...</span>
            )}
          </div>
        </div>

        {/* Time - rendered client-side only */}
        <ClientTime />
      </div>
    </div>
  );
}

function MarketIndex({ 
  symbol, 
  price, 
  change, 
  loading 
}: { 
  symbol: string; 
  price?: number; 
  change?: number;
  loading?: boolean;
}) {
  const isPositive = (change || 0) >= 0;
  
  return (
    <div className="flex items-center gap-2">
      {isPositive ? (
        <TrendingUp className="w-4 h-4 text-emerald-400" />
      ) : (
        <TrendingDown className="w-4 h-4 text-red-400" />
      )}
      <span className="text-white text-sm font-medium">{symbol}</span>
      {loading ? (
        <span className="text-slate-500 text-sm font-mono">--</span>
      ) : (
        <>
          <span className="text-white text-sm font-mono">
            ${formatPrice(price || 0)}
          </span>
          <span className={cn(
            "text-xs font-mono",
            isPositive ? "text-emerald-400" : "text-red-400"
          )}>
            {formatPercent(change || 0)}
          </span>
        </>
      )}
    </div>
  );
}
