'use client';

import { useState, useEffect } from 'react';
import { useWhaleFlows } from '@/hooks/useMarketData';
import { cn, formatCurrency, timeAgo, formatExpiry } from '@/lib/utils';
import { Anchor, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import type { WhaleFlow } from '@/types';

interface WhaleFlowsPanelProps {
  onSelectSymbol?: (symbol: string) => void;
}

export function WhaleFlowsPanel({ onSelectSymbol }: WhaleFlowsPanelProps) {
  const { data: flows, isLoading, refetch, isFetching } = useWhaleFlows();

  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Anchor className="w-4 h-4 text-blue-400" />
          🐋 Whale Flows
          {flows && flows.length > 0 && (
            <span className="text-slate-400 text-sm font-normal">
              ({flows.length})
            </span>
          )}
        </h3>
        
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

      {/* Flows List */}
      <div className="overflow-y-auto max-h-[400px]">
        {isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-20 bg-slate-700 rounded animate-pulse" />
            ))}
          </div>
        ) : !flows || flows.length === 0 ? (
          <div className="p-8 text-center text-slate-500">
            <Anchor className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No whale flows detected</p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {flows.map((flow, idx) => (
              <WhaleFlowCard 
                key={`${flow.symbol}-${flow.strike}-${idx}`} 
                flow={flow} 
                onClick={() => onSelectSymbol?.(flow.symbol)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function WhaleFlowCard({ flow, onClick }: { flow: WhaleFlow; onClick?: () => void }) {
  const [timeDisplay, setTimeDisplay] = useState<string>('');
  const isCall = flow.type === 'CALL';
  
  useEffect(() => {
    if (flow.timestamp) {
      setTimeDisplay(timeAgo(flow.timestamp));
      const interval = setInterval(() => setTimeDisplay(timeAgo(flow.timestamp)), 60000);
      return () => clearInterval(interval);
    }
  }, [flow.timestamp]);
  
  return (
    <div 
      className={cn(
        "p-3 rounded-lg border cursor-pointer hover:brightness-110 transition-all",
        isCall 
          ? "bg-emerald-500/10 border-emerald-500/30" 
          : "bg-red-500/10 border-red-500/30"
      )}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isCall ? (
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          ) : (
            <TrendingDown className="w-4 h-4 text-red-400" />
          )}
          <span className="text-white font-semibold">{flow.symbol}</span>
          <span className={cn(
            "font-mono text-sm",
            isCall ? "text-emerald-400" : "text-red-400"
          )}>
            ${flow.strike} {flow.type}
          </span>
        </div>
        
        <span className="text-xs text-slate-400">
          {flow.expiry ? formatExpiry(flow.expiry) : 'N/A'}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-2 text-xs">
        <div>
          <span className="text-slate-500">Vol/OI</span>
          <span className={cn(
            "ml-1 font-mono",
            flow.volOiRatio >= 3 ? "text-amber-400" : "text-slate-300"
          )}>
            {flow.volOiRatio.toFixed(1)}x
          </span>
        </div>
        <div>
          <span className="text-slate-500">Vol</span>
          <span className="ml-1 text-slate-300 font-mono">
            {flow.volume.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="text-slate-500">Premium</span>
          <span className="ml-1 text-amber-400 font-mono">
            {formatCurrency(flow.premium)}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between mt-2">
        <WhaleScoreMeter score={flow.whaleScore} />
        <span className="text-xs text-slate-500">
          {timeDisplay}
        </span>
      </div>
    </div>
  );
}

function WhaleScoreMeter({ score }: { score: number }) {
  const normalizedScore = Math.min(100, Math.max(0, score / 100)); // Assuming max score ~10000
  const displayScore = Math.min(100, score);
  
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500">Score</span>
      <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={cn(
            "h-full rounded-full transition-all",
            displayScore >= 70 ? "bg-blue-400" : 
            displayScore >= 40 ? "bg-amber-400" : "bg-slate-500"
          )}
          style={{ width: `${normalizedScore * 100}%` }}
        />
      </div>
      <span className={cn(
        "text-xs font-mono",
        displayScore >= 70 ? "text-blue-400" : 
        displayScore >= 40 ? "text-amber-400" : "text-slate-500"
      )}>
        {displayScore}
      </span>
    </div>
  );
}
