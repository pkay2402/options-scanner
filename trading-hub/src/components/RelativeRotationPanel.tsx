'use client';

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { cn } from '@/lib/utils';
import { RefreshCw, TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight } from 'lucide-react';

// Sector ETFs configuration
const SECTOR_ETFS: Record<string, { name: string; color: string }> = {
  'XLK': { name: 'Technology', color: '#3b82f6' },
  'XLF': { name: 'Financials', color: '#22c55e' },
  'XLE': { name: 'Energy', color: '#f97316' },
  'XLV': { name: 'Healthcare', color: '#ec4899' },
  'XLY': { name: 'Consumer Disc', color: '#a855f7' },
  'XLP': { name: 'Consumer Staples', color: '#14b8a6' },
  'XLI': { name: 'Industrials', color: '#eab308' },
  'XLB': { name: 'Materials', color: '#78716c' },
  'XLU': { name: 'Utilities', color: '#06b6d4' },
  'XLRE': { name: 'Real Estate', color: '#8b5cf6' },
  'XLC': { name: 'Communication', color: '#f43f5e' },
};

interface RRGPoint {
  symbol: string;
  rsRatio: number;
  rsMomentum: number;
  quadrant: 'Leading' | 'Weakening' | 'Lagging' | 'Improving';
  direction: 'improving' | 'declining';
  color: string;
  trail: { rsRatio: number; rsMomentum: number }[];
}

type Quadrant = 'Leading' | 'Weakening' | 'Lagging' | 'Improving';

function getQuadrant(rsRatio: number, rsMomentum: number): Quadrant {
  if (rsRatio >= 100 && rsMomentum >= 100) return 'Leading';
  if (rsRatio >= 100 && rsMomentum < 100) return 'Weakening';
  if (rsRatio < 100 && rsMomentum < 100) return 'Lagging';
  return 'Improving';
}

const QUADRANT_COLORS: Record<Quadrant, string> = {
  Leading: '#22c55e',
  Weakening: '#eab308',
  Lagging: '#ef4444',
  Improving: '#3b82f6',
};

async function fetchRRGData(): Promise<RRGPoint[]> {
  // Fetch price data for all sector ETFs and SPY benchmark
  const symbols = [...Object.keys(SECTOR_ETFS), 'SPY'];
  
  try {
    const responses = await Promise.all(
      symbols.map(sym => 
        fetch(`/api/price-history?symbol=${sym}&period=3M`)
          .then(r => r.ok ? r.json() : null)
          .catch(() => null)
      )
    );
    
    const priceData: Record<string, number[]> = {};
    
    symbols.forEach((sym, i) => {
      if (responses[i]?.candles?.length > 0) {
        priceData[sym] = responses[i].candles.map((c: { close: number }) => c.close);
      }
    });
    
    if (!priceData['SPY'] || priceData['SPY'].length < 30) {
      // Return mock data if real data not available
      return generateMockRRGData();
    }
    
    const benchmark = priceData['SPY'];
    const results: RRGPoint[] = [];
    
    for (const [symbol, info] of Object.entries(SECTOR_ETFS)) {
      if (!priceData[symbol] || priceData[symbol].length < 30) continue;
      
      const prices = priceData[symbol];
      const minLen = Math.min(prices.length, benchmark.length);
      
      // Calculate relative strength ratio
      const relativeStrength = prices.slice(-minLen).map((p, i) => 
        p / benchmark.slice(-minLen)[i]
      );
      
      // Calculate RS-Ratio (normalized)
      const rsRatio = calculateNormalizedRS(relativeStrength, 10);
      
      // Calculate RS-Momentum 
      const rsMomentum = calculateRSMomentum(rsRatio, 10);
      
      if (rsRatio.length < 3 || rsMomentum.length < 3) continue;
      
      const currentRatio = rsRatio[rsRatio.length - 1];
      const currentMomentum = rsMomentum[rsMomentum.length - 1];
      const prevRatio = rsRatio[rsRatio.length - 2];
      
      // Build trail (last 3 points)
      const trail = [];
      for (let i = 3; i >= 1; i--) {
        if (rsRatio.length >= i && rsMomentum.length >= i) {
          trail.push({
            rsRatio: rsRatio[rsRatio.length - i],
            rsMomentum: rsMomentum[rsMomentum.length - i],
          });
        }
      }
      
      results.push({
        symbol,
        rsRatio: currentRatio,
        rsMomentum: currentMomentum,
        quadrant: getQuadrant(currentRatio, currentMomentum),
        direction: currentRatio > prevRatio ? 'improving' : 'declining',
        color: info.color,
        trail,
      });
    }
    
    return results.length > 0 ? results : generateMockRRGData();
  } catch (error) {
    console.error('RRG fetch error:', error);
    return generateMockRRGData();
  }
}

function calculateNormalizedRS(relative: number[], period: number): number[] {
  const result: number[] = [];
  
  for (let i = period; i < relative.length; i++) {
    const slice = relative.slice(i - period, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length) || 0.001;
    result.push(100 + ((relative[i] - mean) / std) * 10);
  }
  
  return result;
}

function calculateRSMomentum(rsRatio: number[], period: number): number[] {
  const result: number[] = [];
  
  for (let i = period; i < rsRatio.length; i++) {
    const slice = rsRatio.slice(i - period, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length) || 0.001;
    result.push(100 + ((rsRatio[i] - mean) / std) * 10);
  }
  
  return result;
}

function generateMockRRGData(): RRGPoint[] {
  return Object.entries(SECTOR_ETFS).map(([symbol, info]) => {
    const rsRatio = 85 + Math.random() * 30;
    const rsMomentum = 85 + Math.random() * 30;
    return {
      symbol,
      rsRatio,
      rsMomentum,
      quadrant: getQuadrant(rsRatio, rsMomentum),
      direction: Math.random() > 0.5 ? 'improving' : 'declining',
      color: info.color,
      trail: [
        { rsRatio: rsRatio - 2 + Math.random() * 4, rsMomentum: rsMomentum - 2 + Math.random() * 4 },
        { rsRatio: rsRatio - 1 + Math.random() * 2, rsMomentum: rsMomentum - 1 + Math.random() * 2 },
        { rsRatio, rsMomentum },
      ],
    };
  });
}

function useRRGData() {
  return useQuery({
    queryKey: ['rrg-sectors'],
    queryFn: fetchRRGData,
    staleTime: 5 * 60 * 1000,
    refetchInterval: 10 * 60 * 1000, // Every 10 min - rate limit friendly
  });
}

export function RelativeRotationPanel() {
  const { data, isLoading, refetch, isFetching } = useRRGData();
  const [showTrails, setShowTrails] = useState(true);
  const [hoveredSymbol, setHoveredSymbol] = useState<string | null>(null);

  const quadrantSummary = useMemo(() => {
    if (!data) return { Leading: [], Weakening: [], Lagging: [], Improving: [] };
    
    const summary: Record<Quadrant, RRGPoint[]> = {
      Leading: [],
      Weakening: [],
      Lagging: [],
      Improving: [],
    };
    
    data.forEach(point => {
      summary[point.quadrant].push(point);
    });
    
    return summary;
  }, [data]);

  return (
    <div className="bg-slate-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className="text-xl">🔄</span>
          <h3 className="text-white font-semibold">Relative Rotation</h3>
          <span className="text-slate-500 text-sm">vs SPY</span>
        </div>
        
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
              className="rounded bg-slate-700 border-slate-600"
            />
            Trails
          </label>
          
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="p-1.5 hover:bg-slate-700 rounded"
          >
            <RefreshCw className={cn("w-4 h-4 text-slate-400", isFetching && "animate-spin")} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-[400px]">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : (
          <>
            {/* RRG Chart */}
            <div className="relative h-[350px] bg-slate-900 rounded-lg overflow-hidden">
              <RRGChart 
                data={data || []} 
                showTrails={showTrails}
                hoveredSymbol={hoveredSymbol}
                onHover={setHoveredSymbol}
              />
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-3 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-emerald-500" />
                <span className="text-slate-400">Leading</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-yellow-500" />
                <span className="text-slate-400">Weakening</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-red-500" />
                <span className="text-slate-400">Lagging</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-500" />
                <span className="text-slate-400">Improving</span>
              </div>
            </div>

            {/* Quadrant Summary */}
            <div className="grid grid-cols-4 gap-2 mt-4">
              <QuadrantCard 
                title="Leading" 
                emoji="🟢" 
                color="emerald"
                items={quadrantSummary.Leading}
              />
              <QuadrantCard 
                title="Weakening" 
                emoji="🟡" 
                color="yellow"
                items={quadrantSummary.Weakening}
              />
              <QuadrantCard 
                title="Lagging" 
                emoji="🔴" 
                color="red"
                items={quadrantSummary.Lagging}
              />
              <QuadrantCard 
                title="Improving" 
                emoji="🔵" 
                color="blue"
                items={quadrantSummary.Improving}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function QuadrantCard({ 
  title, 
  emoji, 
  color, 
  items 
}: { 
  title: string; 
  emoji: string; 
  color: 'emerald' | 'yellow' | 'red' | 'blue';
  items: RRGPoint[];
}) {
  const colorClasses = {
    emerald: 'border-emerald-500/30 bg-emerald-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    red: 'border-red-500/30 bg-red-500/5',
    blue: 'border-blue-500/30 bg-blue-500/5',
  };
  
  return (
    <div className={cn("rounded border p-2", colorClasses[color])}>
      <div className="text-xs font-medium text-slate-400 mb-1">
        {emoji} {title}
      </div>
      <div className="space-y-1">
        {items.length === 0 ? (
          <div className="text-slate-600 text-xs">-</div>
        ) : (
          items.slice(0, 4).map(item => (
            <div key={item.symbol} className="flex items-center gap-1 text-xs">
              <span className="font-mono text-white">{item.symbol}</span>
              {item.direction === 'improving' ? (
                <ArrowUpRight className="w-3 h-3 text-emerald-400" />
              ) : (
                <ArrowDownRight className="w-3 h-3 text-red-400" />
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function RRGChart({ 
  data, 
  showTrails,
  hoveredSymbol,
  onHover,
}: { 
  data: RRGPoint[];
  showTrails: boolean;
  hoveredSymbol: string | null;
  onHover: (symbol: string | null) => void;
}) {
  const chartSize = 350;
  const padding = 40;
  const innerSize = chartSize - padding * 2;
  
  // Scale: 80-120 range
  const scale = (value: number) => {
    const normalized = (value - 80) / 40; // 80-120 -> 0-1
    return padding + normalized * innerSize;
  };
  
  const scaleY = (value: number) => {
    const normalized = (value - 80) / 40;
    return chartSize - padding - normalized * innerSize;
  };

  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${chartSize} ${chartSize}`}>
      {/* Quadrant backgrounds */}
      <rect x={scale(100)} y={scaleY(120)} width={innerSize/2} height={innerSize/2} fill="rgba(34, 197, 94, 0.1)" />
      <rect x={scale(100)} y={scaleY(100)} width={innerSize/2} height={innerSize/2} fill="rgba(234, 179, 8, 0.1)" />
      <rect x={scale(80)} y={scaleY(100)} width={innerSize/2} height={innerSize/2} fill="rgba(239, 68, 68, 0.1)" />
      <rect x={scale(80)} y={scaleY(120)} width={innerSize/2} height={innerSize/2} fill="rgba(59, 130, 246, 0.1)" />
      
      {/* Center lines */}
      <line x1={scale(100)} y1={scaleY(80)} x2={scale(100)} y2={scaleY(120)} stroke="#4b5563" strokeWidth="1" strokeDasharray="4,4" />
      <line x1={scale(80)} y1={scaleY(100)} x2={scale(120)} y2={scaleY(100)} stroke="#4b5563" strokeWidth="1" strokeDasharray="4,4" />
      
      {/* Axis labels */}
      <text x={chartSize/2} y={chartSize - 8} fill="#6b7280" fontSize="10" textAnchor="middle">RS-Ratio →</text>
      <text x={12} y={chartSize/2} fill="#6b7280" fontSize="10" textAnchor="middle" transform={`rotate(-90, 12, ${chartSize/2})`}>RS-Momentum →</text>
      
      {/* Grid labels */}
      <text x={scale(80)} y={chartSize - padding + 15} fill="#6b7280" fontSize="9" textAnchor="middle">80</text>
      <text x={scale(100)} y={chartSize - padding + 15} fill="#6b7280" fontSize="9" textAnchor="middle">100</text>
      <text x={scale(120)} y={chartSize - padding + 15} fill="#6b7280" fontSize="9" textAnchor="middle">120</text>
      <text x={padding - 8} y={scaleY(80)} fill="#6b7280" fontSize="9" textAnchor="end" dominantBaseline="middle">80</text>
      <text x={padding - 8} y={scaleY(100)} fill="#6b7280" fontSize="9" textAnchor="end" dominantBaseline="middle">100</text>
      <text x={padding - 8} y={scaleY(120)} fill="#6b7280" fontSize="9" textAnchor="end" dominantBaseline="middle">120</text>
      
      {/* Quadrant labels */}
      <text x={scale(110)} y={scaleY(115)} fill="#22c55e" fontSize="8" opacity="0.7">LEADING</text>
      <text x={scale(110)} y={scaleY(85)} fill="#eab308" fontSize="8" opacity="0.7">WEAKENING</text>
      <text x={scale(85)} y={scaleY(85)} fill="#ef4444" fontSize="8" opacity="0.7">LAGGING</text>
      <text x={scale(85)} y={scaleY(115)} fill="#3b82f6" fontSize="8" opacity="0.7">IMPROVING</text>
      
      {/* Data points */}
      {data.map((point) => {
        const x = scale(point.rsRatio);
        const y = scaleY(point.rsMomentum);
        const isHovered = hoveredSymbol === point.symbol;
        
        return (
          <g key={point.symbol}>
            {/* Trail */}
            {showTrails && point.trail.length > 1 && (
              <path
                d={point.trail.map((t, i) => 
                  `${i === 0 ? 'M' : 'L'} ${scale(t.rsRatio)} ${scaleY(t.rsMomentum)}`
                ).join(' ')}
                fill="none"
                stroke={point.color}
                strokeWidth="2"
                opacity="0.4"
              />
            )}
            
            {/* Point */}
            <circle
              cx={x}
              cy={y}
              r={isHovered ? 10 : 7}
              fill={point.color}
              stroke="white"
              strokeWidth="2"
              style={{ cursor: 'pointer', transition: 'r 0.2s' }}
              onMouseEnter={() => onHover(point.symbol)}
              onMouseLeave={() => onHover(null)}
            />
            
            {/* Label */}
            <text
              x={x + 10}
              y={y - 8}
              fill={point.color}
              fontSize="10"
              fontWeight="bold"
            >
              {point.symbol}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
