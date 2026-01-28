'use client';

import { useEffect, useRef, useState } from 'react';
import { usePriceHistory, useVolumeWalls, useGEXData } from '@/hooks/useMarketData';
import { cn, formatPrice } from '@/lib/utils';
import { BarChart2, Layers, RefreshCw, TrendingUp, TrendingDown } from 'lucide-react';
import type { PriceCandle } from '@/types';

interface PriceChartProps {
  symbol: string;
  className?: string;
}

type Period = '1D' | '5D' | '1M' | '3M';

// Calculate EMA
function calculateEMA(candles: PriceCandle[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  
  candles.forEach((candle, i) => {
    if (i === 0) {
      ema.push(candle.close);
    } else if (i < period) {
      // Simple average for first 'period' candles
      const sum = candles.slice(0, i + 1).reduce((acc, c) => acc + c.close, 0);
      ema.push(sum / (i + 1));
    } else {
      ema.push((candle.close - ema[i - 1]) * multiplier + ema[i - 1]);
    }
  });
  
  return ema;
}

// Calculate VWAP
function calculateVWAP(candles: PriceCandle[]): number[] {
  const vwap: number[] = [];
  let cumulativeTPV = 0;
  let cumulativeVolume = 0;
  
  candles.forEach((candle) => {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    cumulativeTPV += typicalPrice * (candle.volume || 1);
    cumulativeVolume += candle.volume || 1;
    vwap.push(cumulativeTPV / cumulativeVolume);
  });
  
  return vwap;
}

export function PriceChart({ symbol, className }: PriceChartProps) {
  const [period, setPeriod] = useState<Period>('1D');
  const { data: candles, isLoading, refetch, isFetching } = usePriceHistory(symbol, period);
  const { data: volumeWalls, refetch: refetchWalls } = useVolumeWalls(symbol);
  const { data: gexData, refetch: refetchGex } = useGEXData(symbol);
  
  const chartRef = useRef<HTMLDivElement>(null);

  // Auto-refresh handled by React Query hooks (3min for chart, 5min for options)
  // Manual refresh button still available

  // Calculate indicators
  const ema21 = candles ? calculateEMA(candles, 21) : [];
  const vwap = candles ? calculateVWAP(candles) : [];

  // Calculate price stats
  const lastCandle = candles?.[candles.length - 1];
  const firstCandle = candles?.[0];
  const priceChange = lastCandle && firstCandle 
    ? lastCandle.close - firstCandle.open 
    : 0;
  const priceChangePct = firstCandle && firstCandle.open 
    ? (priceChange / firstCandle.open) * 100 
    : 0;
  const high = candles?.reduce((max, c) => Math.max(max, c.high), 0) || 0;
  const low = candles?.reduce((min, c) => Math.min(min, c.low), Infinity) || 0;

  // Current indicator values
  const currentEMA = ema21.length > 0 ? ema21[ema21.length - 1] : null;
  const currentVWAP = vwap.length > 0 ? vwap[vwap.length - 1] : null;

  const handleRefresh = () => {
    refetch();
    refetchWalls();
    refetchGex();
  };

  return (
    <div className={cn("bg-slate-800 rounded-lg overflow-hidden", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-blue-400" />
            <span className="text-white font-bold text-lg">{symbol}</span>
          </div>
          
          {lastCandle && (
            <div className="flex items-center gap-3">
              <span className="text-white font-mono text-lg">
                ${formatPrice(lastCandle.close)}
              </span>
              <span className={cn(
                "font-mono text-sm flex items-center gap-1",
                priceChange >= 0 ? "text-emerald-400" : "text-red-400"
              )}>
                {priceChange >= 0 ? (
                  <TrendingUp className="w-4 h-4" />
                ) : (
                  <TrendingDown className="w-4 h-4" />
                )}
                {priceChange >= 0 ? '+' : ''}{formatPrice(priceChange)} ({priceChangePct >= 0 ? '+' : ''}{priceChangePct.toFixed(2)}%)
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Period Selector */}
          <div className="flex gap-1 bg-slate-700 rounded p-0.5">
            {(['1D', '5D', '1M', '3M'] as Period[]).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={cn(
                  "px-2 py-1 text-xs rounded transition-colors",
                  period === p 
                    ? "bg-blue-500 text-white" 
                    : "text-slate-400 hover:text-white"
                )}
              >
                {p}
              </button>
            ))}
          </div>

          <button
            onClick={handleRefresh}
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

      {/* Chart Area */}
      <div className="p-4">
        <div 
          ref={chartRef} 
          className="h-[300px] bg-slate-900 rounded-lg flex items-center justify-center relative"
        >
          {isLoading ? (
            <div className="flex flex-col items-center gap-2 text-slate-500">
              <RefreshCw className="w-8 h-8 animate-spin" />
              <span>Loading chart...</span>
            </div>
          ) : !candles || candles.length === 0 ? (
            <div className="flex flex-col items-center gap-2 text-slate-500">
              <BarChart2 className="w-8 h-8 opacity-50" />
              <span>No data available</span>
              <span className="text-xs">Connect Schwab API for live data</span>
            </div>
          ) : (
            <CandleChartWithIndicators 
              candles={candles} 
              ema21={ema21}
              vwap={vwap}
            />
          )}
        </div>

        {/* Indicator Legend */}
        {candles && candles.length > 0 && (
          <div className="flex items-center gap-4 mt-2 px-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-yellow-400" />
              <span className="text-xs text-slate-400">
                VWAP: {currentVWAP ? `$${formatPrice(currentVWAP)}` : '--'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-purple-400" />
              <span className="text-xs text-slate-400">
                21 EMA: {currentEMA ? `$${formatPrice(currentEMA)}` : '--'}
              </span>
            </div>
            <span className="text-xs text-slate-600 ml-auto">Auto-refresh: 3min | Schwab API</span>
          </div>
        )}

        {/* Key Levels */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          <LevelCard label="High" value={high} color="text-emerald-400" />
          <LevelCard label="Low" value={low} color="text-red-400" />
          <LevelCard 
            label="Call Wall" 
            value={volumeWalls?.callWall} 
            color="text-emerald-400"
            icon={<Layers className="w-3 h-3" />}
          />
          <LevelCard 
            label="Put Wall" 
            value={volumeWalls?.putWall} 
            color="text-red-400"
            icon={<Layers className="w-3 h-3" />}
          />
        </div>

        {/* GEX Summary */}
        {gexData && (
          <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-sm">Gamma Exposure (GEX)</span>
              <div className="flex items-center gap-4">
                <div>
                  <span className="text-slate-500 text-xs">Total GEX</span>
                  <span className={cn(
                    "ml-2 font-mono text-sm",
                    gexData.totalGex >= 0 ? "text-emerald-400" : "text-red-400"
                  )}>
                    {gexData.totalGex >= 0 ? '+' : ''}{(gexData.totalGex / 1e9).toFixed(2)}B
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 text-xs">Flip</span>
                  <span className="ml-2 font-mono text-sm text-amber-400">
                    ${formatPrice(gexData.flipPrice)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function LevelCard({ 
  label, 
  value, 
  color,
  icon
}: { 
  label: string; 
  value?: number; 
  color: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="bg-slate-700/50 rounded p-2">
      <div className="flex items-center gap-1 text-slate-400 text-xs mb-1">
        {icon}
        {label}
      </div>
      <span className={cn("font-mono text-sm", color)}>
        {value ? `$${formatPrice(value)}` : '--'}
      </span>
    </div>
  );
}

function CandleChartWithIndicators({ 
  candles, 
  ema21,
  vwap 
}: { 
  candles: PriceCandle[];
  ema21: number[];
  vwap: number[];
}) {
  if (!candles.length) return null;

  // Include indicators in price range calculation
  const allPrices = [
    ...candles.map(c => c.high),
    ...candles.map(c => c.low),
    ...ema21.filter(v => v > 0),
    ...vwap.filter(v => v > 0),
  ];
  
  const maxHigh = Math.max(...allPrices);
  const minLow = Math.min(...allPrices);
  const padding = (maxHigh - minLow) * 0.05;
  const range = (maxHigh - minLow + padding * 2) || 1;
  const chartHeight = 280;
  
  const candleWidth = Math.max(2, Math.min(8, 100 / candles.length));
  const gap = candleWidth * 0.3;
  const totalWidth = candles.length * (candleWidth + gap);

  const scaleY = (price: number) => {
    return chartHeight - ((price - minLow + padding) / range) * chartHeight;
  };

  // Build path for line indicators
  const buildLinePath = (values: number[]) => {
    return values
      .map((val, i) => {
        const x = i * (candleWidth + gap) + candleWidth / 2;
        const y = scaleY(val);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };

  const vwapPath = buildLinePath(vwap);
  const ema21Path = buildLinePath(ema21);

  return (
    <svg 
      viewBox={`0 0 ${totalWidth} ${chartHeight}`} 
      className="w-full h-full"
      preserveAspectRatio="none"
    >
      {/* VWAP Line (yellow) */}
      {vwap.length > 0 && (
        <path
          d={vwapPath}
          fill="none"
          stroke="#facc15"
          strokeWidth={1.5}
          opacity={0.8}
        />
      )}
      
      {/* 21 EMA Line (purple) */}
      {ema21.length > 0 && (
        <path
          d={ema21Path}
          fill="none"
          stroke="#a855f7"
          strokeWidth={1.5}
          opacity={0.8}
        />
      )}

      {/* Candles */}
      {candles.map((candle, i) => {
        const x = i * (candleWidth + gap);
        const isGreen = candle.close >= candle.open;
        const bodyTop = scaleY(Math.max(candle.open, candle.close));
        const bodyBottom = scaleY(Math.min(candle.open, candle.close));
        const bodyHeight = Math.max(1, bodyBottom - bodyTop);
        
        return (
          <g key={i}>
            {/* Wick */}
            <line
              x1={x + candleWidth / 2}
              y1={scaleY(candle.high)}
              x2={x + candleWidth / 2}
              y2={scaleY(candle.low)}
              stroke={isGreen ? '#10b981' : '#ef4444'}
              strokeWidth={1}
            />
            {/* Body */}
            <rect
              x={x}
              y={bodyTop}
              width={candleWidth}
              height={bodyHeight}
              fill={isGreen ? '#10b981' : '#ef4444'}
              rx={1}
            />
          </g>
        );
      })}
    </svg>
  );
}
