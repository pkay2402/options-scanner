'use client';

import { useState, useEffect } from 'react';
import { useScannerSignals } from '@/hooks/useMarketData';
import { cn, timeAgo, getSignalBgColor, getSignalColor } from '@/lib/utils';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  Filter,
  RefreshCw
} from 'lucide-react';
import type { ScannerSignal } from '@/types';

type FilterType = 'all' | 'bullish' | 'bearish';

interface ScannerFeedProps {
  onSelectSymbol?: (symbol: string) => void;
}

export function ScannerFeed({ onSelectSymbol }: ScannerFeedProps) {
  const { data: signals, isLoading, refetch, isFetching } = useScannerSignals();
  const [filter, setFilter] = useState<FilterType>('all');

  const filteredSignals = (signals || []).filter(signal => {
    if (filter === 'all') return true;
    return signal.direction === filter;
  });

  const signalCounts = {
    bullish: (signals || []).filter(s => s.direction === 'bullish').length,
    bearish: (signals || []).filter(s => s.direction === 'bearish').length,
  };

  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          Live Scanner
        </h3>
        
        <div className="flex items-center gap-2">
          {/* Filter Pills */}
          <div className="flex gap-1">
            <FilterPill 
              label="All" 
              active={filter === 'all'} 
              onClick={() => setFilter('all')}
            />
            <FilterPill 
              label={`🟢 ${signalCounts.bullish}`}
              active={filter === 'bullish'} 
              onClick={() => setFilter('bullish')}
            />
            <FilterPill 
              label={`🔴 ${signalCounts.bearish}`}
              active={filter === 'bearish'} 
              onClick={() => setFilter('bearish')}
            />
          </div>
          
          {/* Refresh Button */}
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="p-1.5 hover:bg-slate-700 rounded transition-colors"
          >
            <RefreshCw className={cn(
              "w-4 h-4 text-slate-400",
              isFetching && "animate-spin"
            )} />
          </button>
        </div>
      </div>

      {/* Signal List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-16 bg-slate-700 rounded animate-pulse" />
            ))}
          </div>
        ) : filteredSignals.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No signals found</p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {filteredSignals.map((signal) => (
              <SignalCard 
                key={signal.id} 
                signal={signal} 
                onClick={() => onSelectSymbol?.(signal.symbol)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FilterPill({ 
  label, 
  active, 
  onClick 
}: { 
  label: string; 
  active: boolean; 
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-2 py-1 text-xs rounded transition-colors",
        active 
          ? "bg-blue-500/30 text-blue-400" 
          : "bg-slate-700 text-slate-400 hover:text-white"
      )}
    >
      {label}
    </button>
  );
}

function SignalCard({ signal, onClick }: { signal: ScannerSignal; onClick?: () => void }) {
  const [timeDisplay, setTimeDisplay] = useState<string>('');
  
  useEffect(() => {
    setTimeDisplay(timeAgo(signal.timestamp));
    const interval = setInterval(() => setTimeDisplay(timeAgo(signal.timestamp)), 60000);
    return () => clearInterval(interval);
  }, [signal.timestamp]);

  const Icon = signal.direction === 'bullish' 
    ? TrendingUp 
    : signal.direction === 'bearish' 
    ? TrendingDown 
    : Minus;

  return (
    <div 
      className={cn(
        "p-3 rounded-lg border cursor-pointer hover:brightness-110 transition-all",
        getSignalBgColor(signal.direction)
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <Icon className={cn("w-4 h-4", getSignalColor(signal.direction))} />
          <span className="text-white font-semibold">{signal.symbol}</span>
          <SignalTypeBadge type={signal.signalType} />
        </div>
        <span className="text-xs text-slate-400">{timeDisplay}</span>
      </div>
      
      <p className={cn("text-sm mt-1", getSignalColor(signal.direction))}>
        {signal.description}
      </p>
      
      {signal.price > 0 && (
        <p className="text-xs text-slate-400 mt-1 font-mono">
          @ ${signal.price.toFixed(2)}
        </p>
      )}
    </div>
  );
}

function SignalTypeBadge({ type }: { type: ScannerSignal['signalType'] }) {
  const colors: Record<string, string> = {
    MACD: 'bg-purple-500/30 text-purple-400',
    TTM_SQUEEZE: 'bg-orange-500/30 text-orange-400',
    VPB: 'bg-cyan-500/30 text-cyan-400',
    CYCLE: 'bg-pink-500/30 text-pink-400',
    WHALE: 'bg-blue-500/30 text-blue-400',
    ZSCORE: 'bg-indigo-500/30 text-indigo-400',
  };

  return (
    <span className={cn(
      "px-1.5 py-0.5 rounded text-xs",
      colors[type] || 'bg-slate-500/30 text-slate-400'
    )}>
      {type.replace('_', ' ')}
    </span>
  );
}
