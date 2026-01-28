'use client';

import { useState } from 'react';
import { useWatchlist } from '@/hooks/useMarketData';
import { formatPrice, formatPercent, cn } from '@/lib/utils';
import { 
  TrendingUp, 
  TrendingDown, 
  ArrowUpDown, 
  Search,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import type { WatchlistStock } from '@/types';

type SortKey = 'symbol' | 'price' | 'changePct' | 'whaleScore' | 'compositeScore';
type SortDir = 'asc' | 'desc';

interface WatchlistTableProps {
  onSelectSymbol?: (symbol: string) => void;
  selectedSymbol?: string;
}

export function WatchlistTable({ onSelectSymbol, selectedSymbol }: WatchlistTableProps) {
  const { data: stocks, isLoading, error } = useWatchlist();
  const [sortKey, setSortKey] = useState<SortKey>('changePct');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [searchTerm, setSearchTerm] = useState('');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const filteredStocks = (stocks || [])
    .filter(stock => 
      stock.symbol.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      const modifier = sortDir === 'asc' ? 1 : -1;
      
      if (typeof aVal === 'string') {
        return aVal.localeCompare(bVal as string) * modifier;
      }
      return ((aVal as number) - (bVal as number)) * modifier;
    });

  if (error) {
    return (
      <div className="bg-slate-800 rounded-lg p-4">
        <p className="text-red-400 text-sm">Failed to load watchlist</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <h3 className="text-white font-semibold flex items-center gap-2">
          📋 Watchlist
          <span className="text-slate-400 text-sm font-normal">
            ({filteredStocks.length})
          </span>
        </h3>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-slate-700 text-white text-sm pl-9 pr-3 py-1.5 rounded-md w-40 
                       placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-y-auto max-h-[500px]">
        <table className="w-full">
          <thead className="sticky top-0 bg-slate-800">
            <tr className="text-xs text-slate-400 uppercase">
              <SortHeader label="Symbol" sortKey="symbol" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
              <SortHeader label="Price" sortKey="price" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
              <SortHeader label="Change" sortKey="changePct" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
              <SortHeader label="Whale" sortKey="whaleScore" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
              <SortHeader label="Score" sortKey="compositeScore" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 10 }).map((_, i) => (
                <tr key={i} className="border-t border-slate-700/50">
                  <td colSpan={5} className="px-4 py-3">
                    <div className="h-4 bg-slate-700 rounded animate-pulse" />
                  </td>
                </tr>
              ))
            ) : (
              filteredStocks.map((stock) => (
                <StockRow 
                  key={stock.symbol} 
                  stock={stock} 
                  isSelected={stock.symbol === selectedSymbol}
                  onClick={() => onSelectSymbol?.(stock.symbol)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortHeader({ 
  label, 
  sortKey, 
  currentKey, 
  dir, 
  onSort,
  align = 'left'
}: { 
  label: string; 
  sortKey: SortKey; 
  currentKey: SortKey;
  dir: SortDir;
  onSort: (key: SortKey) => void;
  align?: 'left' | 'right';
}) {
  const isActive = currentKey === sortKey;
  
  return (
    <th 
      className={cn(
        "px-4 py-2 cursor-pointer hover:text-white transition-colors",
        align === 'right' && "text-right"
      )}
      onClick={() => onSort(sortKey)}
    >
      <div className={cn("flex items-center gap-1", align === 'right' && "justify-end")}>
        {label}
        {isActive ? (
          dir === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
        ) : (
          <ArrowUpDown className="w-3 h-3 opacity-50" />
        )}
      </div>
    </th>
  );
}

function StockRow({ 
  stock, 
  isSelected,
  onClick 
}: { 
  stock: WatchlistStock;
  isSelected?: boolean;
  onClick?: () => void;
}) {
  const isPositive = stock.changePct >= 0;
  
  return (
    <tr 
      className={cn(
        "border-t border-slate-700/50 hover:bg-slate-700/50 cursor-pointer transition-colors",
        isSelected && "bg-blue-500/20"
      )}
      onClick={onClick}
    >
      <td className="px-4 py-2">
        <div className="flex items-center gap-2">
          {isPositive ? (
            <TrendingUp className="w-3 h-3 text-emerald-400" />
          ) : (
            <TrendingDown className="w-3 h-3 text-red-400" />
          )}
          <span className="text-white font-medium">{stock.symbol}</span>
          {stock.signals.length > 0 && (
            <span className="text-xs text-amber-400">⚡</span>
          )}
        </div>
      </td>
      <td className="px-4 py-2 text-right font-mono text-white text-sm">
        ${formatPrice(stock.price)}
      </td>
      <td className={cn(
        "px-4 py-2 text-right font-mono text-sm",
        isPositive ? "text-emerald-400" : "text-red-400"
      )}>
        {formatPercent(stock.changePct)}
      </td>
      <td className="px-4 py-2 text-right">
        <WhaleIndicator score={stock.whaleScore} />
      </td>
      <td className="px-4 py-2 text-right">
        <ScoreBadge score={stock.compositeScore} />
      </td>
    </tr>
  );
}

function WhaleIndicator({ score }: { score: number }) {
  const level = score >= 80 ? 'high' : score >= 50 ? 'medium' : 'low';
  
  return (
    <div className="flex items-center justify-end gap-1">
      <span className={cn(
        "text-xs font-mono",
        level === 'high' && "text-blue-400",
        level === 'medium' && "text-amber-400",
        level === 'low' && "text-slate-500"
      )}>
        {score}
      </span>
      {level === 'high' && <span>🐋</span>}
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const color = score >= 70 
    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/50" 
    : score >= 40 
    ? "bg-amber-500/20 text-amber-400 border-amber-500/50"
    : "bg-slate-500/20 text-slate-400 border-slate-500/50";

  return (
    <span className={cn(
      "px-2 py-0.5 rounded text-xs font-mono border",
      color
    )}>
      {score}
    </span>
  );
}
